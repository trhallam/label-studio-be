import click
from click_params import IP_ADDRESS, DOMAIN
from validators import url, ip_address, domain, ValidationError
from urllib.parse import urlparse


class DomainIpParamType:
    def __init__(self, value: str):
        self.value = value

    def __repr__(self) -> str:
        return f"<DomainIpParamType: value={self.value}"


class DomainIpParser(click.ParamType):
    name = "DomainIpParamType"

    def convert(self, value, param, ctx):

        try:
            url(value)
            parsed_uri = urlparse(value)
            value = parsed_uri.path if parsed_uri.path else parsed_uri.netloc
        except ValidationError:
            pass

        if ip_address.ipv4(value) or ip_address.ipv6(value):
            return IP_ADDRESS.convert(value, param, ctx)

        if domain(value):
            return DOMAIN.convert(value, param, ctx)

        # assume simple host
        if value:
            return value

        raise click.BadParameter(
            "The input value was not a URL/IP_ADRESS/DOMAIN", ctx, param
        )
