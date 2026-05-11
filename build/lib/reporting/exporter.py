"""Module to export processed data into a beautifully formatted HTML report."""

import json
import webbrowser
from pathlib import Path
from html import escape


def generate_html_template(rows: str, count: int) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Processed Data Inspection</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f172a;
            --text-color: #f8fafc;
            --glass-bg: rgba(30, 41, 59, 0.7);
            --glass-border: rgba(255, 255, 255, 0.1);
            --accent-color: #38bdf8;
            --secondary-text: #94a3b8;
            --table-header-bg: rgba(15, 23, 42, 0.8);
            --row-hover: rgba(56, 189, 248, 0.05);
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: var(--text-color);
            margin: 0;
            padding: 2rem;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        header {{
            text-align: center;
            margin-bottom: 2rem;
            animation: fadeInDown 0.8s ease-out;
        }}

        h1 {{
            font-size: 2.5rem;
            font-weight: 600;
            margin: 0 0 0.5rem 0;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .subtitle {{
            color: var(--secondary-text);
            font-size: 1.1rem;
        }}

        .table-container {{
            width: 100%;
            max-width: 1400px;
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: fadeIn 1s ease-out;
        }}

        .table-wrapper {{
            overflow-x: auto;
            max-height: 75vh;
            overflow-y: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}

        th {{
            background: var(--table-header-bg);
            padding: 1rem 1.5rem;
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--accent-color);
            position: sticky;
            top: 0;
            z-index: 10;
            border-bottom: 1px solid var(--glass-border);
            white-space: nowrap;
        }}

        td {{
            padding: 1.2rem 1.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            line-height: 1.6;
            vertical-align: top;
        }}

        tr {{
            transition: all 0.2s ease;
        }}

        tr:hover {{
            background: var(--row-hover);
            transform: scale(1.001);
        }}

        .col-id {{ width: 5%; color: var(--secondary-text); font-variant-numeric: tabular-nums; }}
        .col-author {{ width: 10%; font-weight: 500; color: #cbd5e1; }}
        .col-query {{ width: 10%; }}
        .col-clean {{ width: 35%; font-size: 1.05rem; }}
        .col-raw {{ width: 40%; color: var(--secondary-text); font-size: 0.9rem; }}

        .tag {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background: rgba(56, 189, 248, 0.1);
            color: var(--accent-color);
            border-radius: 9999px;
            font-size: 0.8rem;
            border: 1px solid rgba(56, 189, 248, 0.2);
            white-space: nowrap;
        }}

        /* Custom Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: rgba(15, 23, 42, 0.5);
        }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.3);
        }}

        @keyframes fadeInDown {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
    </style>
</head>
<body>

    <header>
        <h1>Data Inspection</h1>
        <div class="subtitle">Manual review of <b>{count}</b> preprocessed posts</div>
    </header>

    <div class="table-container">
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr>
                        <th class="col-id">#</th>
                        <th class="col-author">Author</th>
                        <th class="col-query">Query</th>
                        <th class="col-clean">Cleaned Content</th>
                        <th class="col-raw">Original Content</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
    </div>

</body>
</html>"""


def export_processed_to_html(input_path: Path, output_dir: Path) -> Path:
    """Read a processed JSONL file and generate a beautiful HTML table, then open it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{input_path.stem}_inspection.html"

    rows = []
    count = 0

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            post = json.loads(line)
            count += 1
            
            author = escape(post.get("username") or "Unknown")
            query = escape(post.get("query") or "N/A")
            cleaned = escape(post.get("cleaned_content") or "")
            raw = escape(post.get("content") or "")

            row_html = f"""
            <tr>
                <td class="col-id">{count}</td>
                <td class="col-author">@{author}</td>
                <td class="col-query"><span class="tag">{query}</span></td>
                <td class="col-clean">{cleaned}</td>
                <td class="col-raw">{raw}</td>
            </tr>
            """
            rows.append(row_html)

    # Generate complete HTML
    html_content = generate_html_template("".join(rows), count)

    # Write to file
    with html_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    # Open in browser
    webbrowser.open(f"file://{html_path.absolute()}")
    
    return html_path
