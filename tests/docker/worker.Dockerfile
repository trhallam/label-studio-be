FROM python:3

WORKDIR /usr/src/app

COPY --from=project . .

RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install .

RUN cat <<EOF >> /usr/sbin/docker_entrypoint.sh
#!/bin/bash
rq worker \
  -v \
  --logging_level INFO \
  --max-idle-time 9999999999 \
  --worker-class 'label_studio_berq.worker.LabelStudioBEWorker' \
  -u 'redis://redis:6379' \
  $@
EOF

RUN chmod +x /usr/sbin/docker_entrypoint.sh

ENTRYPOINT [ "/usr/sbin/docker_entrypoint.sh" ]
