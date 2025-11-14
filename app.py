#!/usr/bin/env python3
"""
Minimal Gradio UI for TorchTalk chatbot.

Usage:
    python app.py --index-path ./index --vllm-server http://localhost:8000
"""

import argparse
import logging
import sys
from pathlib import Path

import gradio as gr

from torchtalk.engine import ConversationEngine

log = logging.getLogger(__name__)


def create_app(
    index_path: str,
    vllm_server: str = "http://localhost:8000",
    model_name: str = "meta-llama/llama-4-maverick",
) -> gr.Blocks:
    """
    Create Gradio chat interface.

    Args:
        index_path: Path to persisted index
        vllm_server: vLLM server URL
        model_name: Model name

    Returns:
        Gradio Blocks app
    """
    # Initialize engine eagerly
    log.info("Initializing conversation engine...")
    engine = ConversationEngine(
        index_path=index_path,
        vllm_server=vllm_server,
        model_name=model_name,
    )

    def chat_fn(message: str, history):
        """Chat function for Gradio - returns updated history"""
        if not message.strip():
            return history

        try:
            reply = engine.chat(message)
            history = history + [[message, reply]]
            return history
        except Exception as e:
            log.error(f"Chat error: {e}", exc_info=True)
            history = history + [[message, f"Error: {e}"]]
            return history

    def reset_fn():
        """Reset conversation"""
        engine.reset()
        return []

    # Create Gradio interface
    with gr.Blocks(title="TorchTalk - PyTorch Codebase Assistant", theme="default") as app:
        gr.Markdown(
            """
            # TorchTalk 🔥

            Ask questions about the PyTorch codebase with cross-language tracing (Python ↔ C++ ↔ CUDA).

            **Features:**
            - Conversation memory for follow-up questions
            - Graph-enhanced retrieval with binding information
            - 1M context window support
            """
        )

        chatbot = gr.Chatbot(
            label="Chat History",
            height=500,
            show_copy_button=True,
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Your question",
                placeholder="Ask about PyTorch internals...",
                scale=4,
                lines=2,
            )
            submit = gr.Button("Send", scale=1, variant="primary")

        with gr.Row():
            clear = gr.Button("Clear Chat")

        # Event handlers
        submit.click(
            chat_fn,
            inputs=[msg, chatbot],
            outputs=[chatbot],
        ).then(
            lambda: "",
            outputs=[msg],
        )

        msg.submit(
            chat_fn,
            inputs=[msg, chatbot],
            outputs=[chatbot],
        ).then(
            lambda: "",
            outputs=[msg],
        )

        clear.click(
            reset_fn,
            outputs=[chatbot],
        )

    return app


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="TorchTalk Gradio UI")
    parser.add_argument(
        "--index-path",
        required=True,
        help="Path to persisted LlamaIndex storage"
    )
    parser.add_argument(
        "--vllm-server",
        default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/llama-4-maverick",
        help="Model name (default: meta-llama/llama-4-maverick)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on (default: 7860)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Verify index exists
    if not Path(args.index_path).exists():
        log.error(f"Index not found at {args.index_path}")
        log.error("Run indexing first: python -m torchtalk.indexing.graph_enhanced_indexer")
        sys.exit(1)

    # Create and launch app
    log.info("Creating Gradio app...")
    app = create_app(
        index_path=args.index_path,
        vllm_server=args.vllm_server,
        model_name=args.model_name,
    )

    log.info(f"Launching on {args.host}:{args.port}...")
    app.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
