from social_listening_pipeline.cli import build_parser


def test_parser_includes_show_config_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["--show-config"])
    assert args.show_config is True
