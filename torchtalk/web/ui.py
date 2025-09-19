import gradio as gr
import requests

TORCHTALK_API = "http://localhost:8001"


def chat_with_torchtalk(message, history):
    try:
        response = requests.post(
            f"{TORCHTALK_API}/chat",
            json={"message": message},
            timeout=300
        )
        response.raise_for_status()

        result = response.json()
        ai_response = result["response"]

        # If empty response, show helpful message
        if not ai_response.strip():
            ai_response = (
                "The model didn't respond sorry :( )"
            )

        return ai_response
    except requests.exceptions.RequestException as e:
        return f"Error connecting to TorchTalk API: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="TorchTalk POC") as demo:
    gr.Markdown("# TorchTalk - Chat with PyTorch Codebase")
    gr.Markdown(
        "Ask questions about PyTorch internals, autograd, neural networks, "
        "and more!"
    )
    chatbot = gr.ChatInterface(
        fn=chat_with_torchtalk,
        title="PyTorch Expert Assistant",
        description="Powered by PyTorch compendium + AI",
        examples=[
            "What is autograd?",
            "How does torch.nn.Module work?",
            "Explain PyTorch tensors",
            "What are the main PyTorch optimizers?"
        ],
        retry_btn=None,
        undo_btn=None,
        clear_btn="Clear Chat"
    )



if __name__ == "__main__":
    print("Starting TorchTalk UI...")
    print(
        "Make sure your FastAPI server is running on "
        "http://localhost:8001"
    )
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

