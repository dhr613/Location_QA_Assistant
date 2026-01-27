import base64
import os
import uuid

import streamlit as st

from react_agent.router import workflow

# 设置页面配置
st.set_page_config(page_title="一起吃吃喝喝", page_icon="🍔", layout="wide")


# 设置背景图片
def set_background_image(image_path: str):
    """设置页面背景图片"""
    try:
        # 读取图片文件并转换为 base64
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode()

        # 注入 CSS 样式
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{img_base64}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            /* 添加半透明遮罩层以提高内容可读性 */
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(255, 255, 255, 0.85);
                z-index: -1;
            }}
            /* 确保主要内容区域有更好的可读性 */
            .main .block-container {{
                background-color: rgba(255, 255, 255, 0.95);
                padding: 2rem;
                border-radius: 10px;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.warning(f"背景图片未找到: {image_path}")
    except Exception as e:
        st.warning(f"设置背景图片时出错: {str(e)}")


# 设置背景
set_background_image("imgs/1.jpg")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 页面标题
st.title("🍔 一起吃吃喝喝吧！")
st.markdown("---")

# 侧边栏说明
with st.sidebar:
    st.header("📖 使用说明")
    st.markdown("""
    这是一个基于 LangGraph 和 LangChain 的智能问答系统，
    可以帮助您：
    - 🍽️ 查找附近的美食餐厅
    - 🎮 寻找娱乐场所
    - 🗺️ 规划出行路线
    - 📍 搜索周边地点
    
    只需在下方输入您的问题，系统会为您提供详细的答案！
    """)
    st.markdown("---")
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 异步生成器函数：处理用户查询并流式输出
async def process_query_stream(query: str, thread_id: str):
    """处理查询并流式返回结果"""
    config = {
        "configurable": {
            "thread_id": thread_id,
        },
    }

    # 准备输入
    input_data = {"query": query}

    # 流式执行 workflow
    final_answer = ""
    seen_statuses = set()

    try:
        async for event in workflow.astream(input_data, config=config):
            # 处理每个事件
            for node_name, node_data in event.items():
                if node_name == "classify" and "classifications" in node_data:
                    classifications = node_data["classifications"]
                    if classifications:
                        sources = [
                            c.source.replace("_", " ").title() for c in classifications
                        ]
                        status_key = f"classify_{','.join(sources)}"
                        if status_key not in seen_statuses:
                            seen_statuses.add(status_key)
                            yield f"🔍 正在分析您的问题，将使用以下服务：{', '.join(sources)}\n\n"

                elif node_name == "around_search_agent":
                    if "results" in node_data:
                        status_key = "around_search"
                        if status_key not in seen_statuses:
                            seen_statuses.add(status_key)
                            yield "⏳ 正在搜索周边地点...\n\n"

                elif node_name == "path_planning_agent":
                    if "results" in node_data:
                        status_key = "path_planning"
                        if status_key not in seen_statuses:
                            seen_statuses.add(status_key)
                            yield "🗺️ 正在规划路线...\n\n"

                elif node_name == "synthesize" and "final_answer" in node_data:
                    final_answer = node_data["final_answer"]

        # 返回最终答案
        if final_answer:
            yield f"\n**回答：**\n\n{final_answer}"
        else:
            yield "\n抱歉，未能生成完整的答案。"

    except Exception as e:
        yield f"\n❌ 处理过程中发生错误：{str(e)}"


# 用户输入
if prompt := st.chat_input("请输入您的问题，例如：成都武侯区有哪些好吃的火锅店？"):
    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示助手回复 - 使用 st.write_stream 简化流式输出
    with st.chat_message("assistant"):
        # 使用 write_stream 自动处理异步生成器的流式输出
        full_response = st.write_stream(
            process_query_stream(prompt, st.session_state.thread_id)
        )

        # 添加助手回复到历史
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
