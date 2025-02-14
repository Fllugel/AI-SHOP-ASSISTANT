import gradio as gr
import random
import string
from chat import process_message

def handle_chat(user_input, chat_history, user_id):
    if not user_id:
        user_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    response = process_message(user_id, user_input)

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})

    return chat_history, "", user_id


def launch_gradio_interface():

    with gr.Blocks(title="Асистент Аврора") as interface:
        user_id = gr.State("")

        with gr.Column():
            gr.Markdown("# Асистент Аврора", elem_id="title")

            chatbot = gr.Chatbot(
                elem_id="chatbot",
                bubble_full_width=False,
                height=800,
                type="messages"
            )

            with gr.Row():
                input_txt = gr.Textbox(
                    placeholder="Напишіть ваше повідомлення...",
                    scale=7,
                    container=False
                )
                submit_btn = gr.Button("Надіслати", variant="primary", scale=1)

        submit_btn.click(
            fn=handle_chat,
            inputs=[input_txt, chatbot, user_id],
            outputs=[chatbot, input_txt, user_id]
        )

        input_txt.submit(
            fn=handle_chat,
            inputs=[input_txt, chatbot, user_id],
            outputs=[chatbot, input_txt, user_id]
        )

    interface.launch()
