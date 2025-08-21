import asyncio
import datetime
import json
import time
import os
from decimal import Decimal
from fastmcp import Client as McpClient
import dashscope
from dateutil.relativedelta import relativedelta
from typing import Annotated, List
from langchain_core.tools import tool
import akshare as ak
from langchain_experimental.graph_transformers.llm import system_prompt
from volcenginesdkarkruntime import Ark
import streamlit_nested_layout

from web.utils.analysis_runner import run_stock_analysis


# 个股分析工具
@tool
def stock_analysis(
        stock_symbols: Annotated[List[str], "股票代码列表"],
        market_type: Annotated[str, "市场类型"]
) -> str:
    """
    这是一个批量分析多个个股的工具。根据股票代码列表和市场类型，逐个分析个股行情并汇总结果。
    Args:
        stock_symbol (str): 股票代码，如 AAPL, TSM
        market_type (str): 市场类型，如 A股，美股
    Returns:
        str: 包含拼接后的个股分析结果的格式化报告
    """
    report_list = []

    for symbol in stock_symbols:
        try:
            result = run_stock_analysis(
                stock_symbol=symbol,
                analysis_date=str(datetime.date.today()),
                analysts=['fundamentals'],
                research_depth=1,
                llm_provider="dashscope",
                llm_model="qwen-plus-latest",
                market_type=market_type,
                progress_callback=None
            )
            report_list.append(f"【{symbol} 股票分析报告】\n{result}\n")
        except Exception as e:
            report_list.append(f"【{symbol} 股票分析失败】\n错误信息: {e}\n")

    return "\n".join(report_list)

def run_fund_analysis(fund_symbol):
    # 构建报告头
    result = f"【基金代码】: {fund_symbol}\n"

    # 1. 基本数据
    try:
        basic_info = ak.fund_individual_basic_info_xq(symbol=fund_symbol)
        result += "【基本数据】:\n" + basic_info.to_string(index=False) + "\n\n"
        time.sleep(1)
    except Exception as e:
        result += f"【基本数据】获取失败: {str(e)}\n\n"

    # 2. 基金评级
    try:
        fund_rating_all_df = ak.fund_rating_all()
        result += "【基金评级】:\n" + fund_rating_all_df[
            fund_rating_all_df['代码'] == fund_symbol
            ].to_string(index=False) + "\n\n"
        time.sleep(1)
    except Exception as e:
        result += f"【基金评级】获取失败: {str(e)}\n\n"

    # 3. 业绩表现（前5条）
    try:
        achievement = ak.fund_individual_achievement_xq(symbol=fund_symbol)
        result += "【业绩表现】:\n" + achievement.head(5).to_string(index=False) + "\n\n"
        time.sleep(1)
    except Exception as e:
        result += f"【业绩表现】获取失败: {str(e)}\n\n"

    # 4. 净值估算（特殊处理全量请求）
    try:
        fund_value_df = ak.fund_value_estimation_em(symbol="全部")
        result += "【净值估算】:\n" + fund_value_df[
            fund_value_df['基金代码'] == fund_symbol
            ].to_string(index=False) + "\n\n"
        time.sleep(1)
    except Exception as e:
        result += f"【净值估算】获取失败: {str(e)}\n\n"

    # 5. 数据分析
    try:
        analysis = ak.fund_individual_analysis_xq(symbol=fund_symbol)
        result += "【数据分析】:\n" + analysis.to_string(index=False) + "\n\n"
        time.sleep(1)
    except Exception as e:
        result += f"【数据分析】获取失败: {str(e)}\n\n"

    # 6. 盈利概率
    try:
        profit_prob = ak.fund_individual_profit_probability_xq(symbol=fund_symbol)
        result += "【盈利概率】:\n" + profit_prob.to_string(index=False) + "\n\n"
        time.sleep(1)
    except Exception as e:
        result += f"【盈利概率】获取失败: {str(e)}\n\n"

    # 7. 持仓资产比例
    try:
        detail_hold = ak.fund_individual_detail_hold_xq(symbol=fund_symbol)
        result += "【持仓资产比例】:\n" + detail_hold.to_string(index=False) + "\n\n"
        time.sleep(1)
    except Exception as e:
        result += f"【持仓资产比例】获取失败: {str(e)}\n\n"

    # 8. 行业配置（2025年数据）
    try:
        industry_alloc = ak.fund_portfolio_industry_allocation_em(symbol=fund_symbol, date="2025")
        result += "【行业配置】:\n" + industry_alloc.to_string(index=False) + "\n\n"
        time.sleep(1)
    except Exception as e:
        result += f"【行业配置】获取失败: {str(e)}\n\n"

    # 9. 基金持仓（2025年数据）
    try:
        portfolio_hold = ak.fund_portfolio_hold_em(symbol=fund_symbol, date="2025")
        result += "【基金持仓】:\n" + portfolio_hold.to_string(index=False) + "\n"
        time.sleep(1)
    except Exception as e:
        result += f"【基金持仓】获取失败: {str(e)}\n"

    print(result)

    system_message = (
        f"你是一位专业的基金基本面分析师。\n"
        f"任务：对（基金代码：{fund_symbol}）进行全面基本面分析\n"
        "📊 **强制要求：**\n"
        "按以下框架输出结构化报告：\n\n"

        "### 一、基金产品基础分析\n"
        "- **基金公司实力**：管理规模排名、权益投资能力评级、风控体系完善度\n"
        "- **基金经理**：从业年限、历史年化回报、最大回撤控制能力（近3年）、投资风格稳定性\n"
        "- **产品特性**：基金类型(股票/混合/债券)、运作方式(开放式/封闭式)、规模变动趋势(警惕＜1亿清盘风险)\n"
        "- **费率结构**：管理费+托管费总成本、浮动费率机制(如有)、申购赎回费率\n\n"

        "### 二、风险收益特征分析\n"
        "- **核心指标**：\n"
        "  • 夏普比率(＞1为优)、卡玛比率(年化收益/最大回撤，＞0.5合格)\n"
        "  • 波动率(同类排名后30%为佳)、下行捕获率(＜100%表明抗跌)\n"
        "- **极端风险控制**：\n"
        "  • 最大回撤率(数值绝对值越小越好)及修复时长\n"
        "  • 股灾/熊市期间表现(如2022年回撤幅度 vs 沪深300)\n\n"

        "### 三、长期业绩评估\n"
        "- **收益维度**：\n"
        "  • 3年/5年年化收益率(需扣除费率)、超额收益(Alpha)\n"
        "  • 业绩持续性：每年排名同类前50%的年份占比\n"
        "- **基准对比**：\n"
        "  • 滚动3年跑赢业绩比较基准的概率\n"
        "  • 不同市场环境适应性(如2023成长牛 vs 2024价值修复行情表现)\n\n"

        "### 四、综合价值评估\n"
        "- **持仓穿透估值**：\n"
        "  • 股票部分：前十大重仓股PE/PB分位数(行业调整后)\n"
        "  • 债券部分：信用债利差水平、利率债久期风险\n"
        "- **组合性价比**：\n"
        "  • 股债净资产比价(E/P - 10年国债收益率)\n"
        "  • 场内基金需分析折溢价率(＞1%警惕高估)\n"
        f"- **绝对价值锚点**：给出合理净值区间依据：\n"
        "  当前净值水平 vs 历史波动区间(30%分位以下为低估)\n\n"

        "### 五、投资决策建议\n"
        "- **建议逻辑**：\n"
        "  • 综合夏普比率＞1.2+卡玛比率＞0.7+净值处30%分位→'买入'\n"
        "  • 规模激增(＞100亿)+重仓股估值＞70%分位→'减持'\n"
        "- **强制输出**：中文操作建议(买入/增持/持有/减持/卖出)\n"

        "🚫 **禁止事项**：\n"
        "- 禁止假设数据\n"
        "- 禁止使用英文建议(buy/sell/hold)\n"
    )

    user_prompt = (f"你现在拥有以下基金的真实数据，请严格依赖真实数据（注意！每条数据必须强制利用到来进行分析），"
                   f"绝不编造其他数据，对（基金代码：{fund_symbol}）进行全面分析，给出非常详细格式化的报告:\n")
    user_prompt += result

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_prompt}
    ]
    response = dashscope.Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model="qwen-plus-latest",
        # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages,
        result_format='message'
    )
    print(response.output.choices[0].message.content)

    return response.output.choices[0].message.content

# 多个公募基金分析工具
@tool
def fund_analysis(
        fund_symbols: Annotated[List[str], "基金代码列表"]
) -> str:
    """
    这是一个专门分析多个公募基金的工具，可以根据基金代码分析多个公募基金的行情。
    Args:
        fund_symbol (str): 基金代码
    Returns:
        str: 包含多个公募基金分析结果的格式化报告
    """
    report_list = []

    for symbol in fund_symbols:
        try:
            result = run_fund_analysis(
                fund_symbol=symbol
            )
            report_list.append(f"【{symbol} 基金分析报告】\n{result}\n")
        except Exception as e:
            report_list.append(f"【{symbol} 基金分析失败】\n错误信息: {e}\n")

    return "\n".join(report_list)
def _format_row(row: dict) -> str:
    """把单条记录格式化成一句话"""
    return (
        f"{row['机构']} 在 {row['指标日期']} "
        f"的 {row['指标名']}（指标ID={row['指标ID']}）"
        f" 余额为 {row['指标值']} 元"
    )
MODEL = "doubao-seed-1-6-flash-250715"
ARK_API_KEY = "ac143c75-41a0-4090-a068-2998091929f4"

PROMPT = (
    "你是一位专业的机构情况分析师，你很擅长:提炼、总结、分析机构金融业务的关键信息，掌握代发指标的分析逻辑，并结合指标数据，公正客观的描述。"
        "任务：分析机构情况记录（{data}）。🔴立即调用 get_deposit_by_id 工具。"
        "整体经营情况;从指标日期、指标值等说明本机构整体经营情况。"
        "📊要求输出一份报告，报告分为三部分，包括整体经营情况、具体分析、存在问题和建议。"
        "具体分析进行数据对比，可以根据数据从下辖机构、较同期、较上月、较上日等角度进行对比。"
        "存在问题和建议:针对上述分析数据，如果有特别异常的数据可以说明，并给出可行的建议。"
        "必须严格依据数据中的信息进行分析，不能虚构或假设数据，必须明确引用数据中的信息，确保数据的真实性和准确性。"
        "逻辑要清晰，语句要通顺，逐步思考，深度挖掘数据体现的企业经营能力，用词专业，客观中立。"
        "🚫禁止假设、英文、编造；✅必须用中文。"
)

async def deposit_analyze(deposit_id: Annotated[str,"机构名称"]) -> str:
    """
       这是一个专门分析机构运营情况的工具，可以根据机构名称查询机构其他指标信息分析机构运营情况。
       Args:
           deposit_id (str): 机构名称
       Returns:
           str: 包含机构运营情况分析结果的格式化报告
       """
    async with McpClient("http://localhost:3000/sse") as mcp:
        raw = await mcp.call_tool("get_deposit_by_id", {"deposit_id": deposit_id})
        rows = json.loads(raw.content[0].text)["results"]
        print(rows)
        # 将 Decimal 转成 float，防止大模型解析报错
        for r in rows:
            if isinstance(r.get("存款数"), Decimal):
                r["存款数"] = float(r["存款数"])

        # 拼成多行文字
        data_summary = "\n".join(_format_row(r) for r in rows)

        from volcenginesdkarkruntime import Ark
        ark = Ark(api_key=ARK_API_KEY)
        resp = ark.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT.format(data=data_summary)}],
            temperature=0.1
        )
        return resp.choices[0].message.content or "分析失败"

# async def main():
#     deposit_id = "思迈特银行"
#     analysis = await deposit_analyze(deposit_id)  # 复用 _analyze
#     print("\n【存款分析】\n", analysis)
# if __name__ == "__main__":
#     asyncio.run(main())