from collector.yahoo_realtime import ScrapeRecord, deduplicate_records, extract_records_from_html


def test_deduplicate_records_removes_duplicates() -> None:
    record = ScrapeRecord(
        query='site:x.com "openai"',
        keyword="openai",
        username="user1",
        timestamp=None,
        content="sample",
        url="https://x.com/user1/status/1",
        media_urls=[],
        fetched_at="2026-01-01T00:00:00+00:00",
        scroll_index=0,
    )
    deduped = deduplicate_records([record, record])
    assert len(deduped) == 1


def test_extract_records_from_html_extracts_x_links() -> None:
    html = """
    <html><body>
      <div id="web">
        <li>
          <a href="https://x.com/tester/status/123">Tweet</a>
          <p>Hello from Yahoo snippet</p>
        </li>
      </div>
    </body></html>
    """
    records = extract_records_from_html(
        html,
        keyword="test",
        query='site:x.com "test"',
        scroll_index=1,
        fetched_at="2026-01-01T00:00:00+00:00",
    )
    assert len(records) == 1
    assert records[0].url == "https://x.com/tester/status/123"
    assert records[0].username == "tester"
    assert records[0].scroll_index == 1
