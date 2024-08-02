import os
import argparse
import json
import logging
import logging.config

logging.config.dictConfig(
    {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": os.getenv("LOG_LEVEL"),
                "stream": "ext://sys.stdout",
                "formatter": "standard",
            }
        },
        "root": {
            "level": os.getenv("LOG_LEVEL"),
            "handlers": ["console"],
            "propagate": True,
        },
    }
)

from label_studio_ml.api import init_app
from label_studio_be.models import models


_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def get_kwargs_from_config(config_path=_DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        return dict()
    with open(config_path) as f:
        config = json.load(f)
    assert isinstance(config, dict)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label studio")
    parser.add_argument(
        "-p", "--port", dest="port", type=int, default=9090, help="Server port"
    )
    parser.add_argument(
        "--host", dest="host", type=str, default="0.0.0.0", help="Server host"
    )
    parser.add_argument(
        "--kwargs",
        "--with",
        dest="kwargs",
        metavar="KEY=VAL",
        nargs="+",
        type=lambda kv: kv.split("="),
        help="Additional LabelStudioMLBase model initialization kwargs",
    )
    parser.add_argument(
        "-d", "--debug", dest="debug", action="store_true", help="Switch debug mode"
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level",
    )
    parser.add_argument(
        "--model-dir",
        dest="model_dir",
        default=os.path.dirname(__file__),
        help="Directory where models are stored (relative to the project directory)",
    )
    parser.add_argument(
        "--model-name",
        dest="model_name",
        default=os.environ.get("LSBE_MODEL_NAME"),
        choices=models.keys(),
        help="The ML model to load for this BE.",
    )
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="Validate model instance before launching server",
    )
    parser.add_argument(
        "--basic-auth-user",
        default=os.environ.get("LSBE_BASIC_AUTH_USER", None),
        help="Basic auth user",
    )

    parser.add_argument(
        "--basic-auth-pass",
        default=os.environ.get("LSBE_BASIC_AUTH_PASS", None),
        help="Basic auth pass",
    )

    args = parser.parse_args()

    # setup logging level
    if args.log_level:
        logging.root.setLevel(args.log_level)

    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def parse_kwargs():
        param = dict()
        for k, v in args.kwargs:
            if v.isdigit():
                param[k] = int(v)
            elif v == "True" or v == "true":
                param[k] = True
            elif v == "False" or v == "false":
                param[k] = False
            elif isfloat(v):
                param[k] = float(v)
            else:
                param[k] = v
        return param

    kwargs = get_kwargs_from_config()

    if args.kwargs:
        kwargs.update(parse_kwargs())

    ml_model = models[args.model_name]

    if args.check:
        print('Check "' + ml_model.__name__ + '" instance creation..')
        model = ml_model(**kwargs)

    app = init_app(
        model_class=ml_model,
        basic_auth_user=args.basic_auth_user,
        basic_auth_pass=args.basic_auth_pass,
    )

    app.run(host=args.host, port=args.port, debug=args.debug)

else:
    # for uWSGI use

    model_name = os.environ.get("LSBE_MODEL_NAME")
    try:
        ml_model = models[model_name]
    except KeyError:
        print(f"env var LSBE_MODEL_NAME must be one of {models.keys()}")
    app = init_app(model_class=ml_model)
