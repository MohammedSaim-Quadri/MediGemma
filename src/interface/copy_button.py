import html
import streamlit.components.v1 as components


def render_copy_button(text: str, height: int = 36):
    """Render a small copy-to-clipboard button using an iframe with JS."""
    escaped = html.escape(text).replace("`", "\\`").replace("${", "\\${")
    components.html(f"""
    <style>
      .copy-wrap {{ display:flex; justify-content:flex-end; }}
      .copy-btn {{
        background:none; border:1px solid #ddd; border-radius:6px;
        cursor:pointer; padding:2px 8px; font-size:14px; color:#888;
        transition: all 0.2s;
      }}
      .copy-btn:hover {{ background:#f0f0f0; color:#333; }}
      .copy-btn.copied {{ color:#22c55e; border-color:#22c55e; }}
    </style>
    <div class="copy-wrap">
      <button class="copy-btn" onclick="
        navigator.clipboard.writeText(`{escaped}`).then(() => {{
          this.innerHTML='&#x2705; Copied';
          this.classList.add('copied');
          setTimeout(() => {{ this.innerHTML='&#x1F4CB; Copy'; this.classList.remove('copied'); }}, 1500);
        }})
      ">&#x1F4CB; Copy</button>
    </div>
    """, height=height)
