import pytest
from click_params import IP_ADDRESS, DOMAIN
from label_studio_berq.typer import DomainIpParser


@pytest.mark.parametrize(
    "value,ans",
    (
        ("http://url.com", DOMAIN("url.com")),
        ("https://url.com.au", DOMAIN("url.com.au")),
        ("redis://server", "server"),
        ("http://127.0.0.1", IP_ADDRESS("127.0.0.1")),
        ("127.0.0.1", IP_ADDRESS("127.0.0.1")),
        ("server", "server"),
        ("server.server.local", DOMAIN("server.server.local")),
    ),
)
def test_DomainIpParser(value, ans):
    parser = DomainIpParser()

    assert ans == parser.convert(value, None, None)
