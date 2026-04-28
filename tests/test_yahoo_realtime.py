from collector.yahoo_realtime import (
    ScrapeRecord,
    deduplicate_records,
    extract_card_record,
    parse_tweet_id_from_data_cl,
    validate_chrome_setup,
)


def test_deduplicate_records_removes_duplicates() -> None:
    record = ScrapeRecord(
        query='site:x.com "openai"',
        keyword="openai",
        username="user1",
        timestamp=None,
        content="sample",
        url="https://search.yahoo.co.jp/realtime/search/tweet/1?detail=1",
        media_urls=[],
        fetched_at="2026-01-01T00:00:00+00:00",
        scroll_index=0,
    )
    deduped = deduplicate_records([record, record])
    assert len(deduped) == 1


def test_parse_tweet_id_from_data_cl() -> None:
    data_cl = "_cl_vmodule:rl;twid:1900000123456789;_cl_link:tw"
    assert parse_tweet_id_from_data_cl(data_cl) == "1900000123456789"


def test_extract_card_record_from_yahoo_realtime_card_html() -> None:
    html = """
    <article class="Tweet_Tweet__sna2i">
      <div class="Tweet_info__bBT3t">
        <p class="Tweet_author__h0pGD">
          <a class="Tweet_authorID__JKhEb">@tester</a>
        </p>
      </div>
      <p>最高のドラマでした</p>
      <time datetime="2026-04-28T12:00:00+09:00"></time>
      <a data-cl-params="_cl_vmodule:rl;twid:1900000999999999;_cl_link:tw"></a>
    </article>
    """

    record, tweet_id = extract_card_record(
        card_html=html,
        keyword="25時赤坂で",
        query="25時赤坂で",
        fetched_at="2026-01-01T00:00:00+00:00",
        scroll_index=2,
    )

    assert tweet_id == "1900000999999999"
    assert record is not None
    assert record.username == "tester"
    assert record.timestamp == "2026-04-28T12:00:00+09:00"
    assert record.scroll_index == 2
    assert "1900000999999999" in record.url


def test_validate_chrome_setup_raises_when_browser_missing(monkeypatch) -> None:
    monkeypatch.setattr("collector.yahoo_realtime.find_browser_binary", lambda: None)

    try:
        validate_chrome_setup()
        assert False, "validate_chrome_setup should raise when browser is missing"
    except RuntimeError as exc:
        assert "Chrome/Chromium browser not found" in str(exc)
