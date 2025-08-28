# Tweet Agent worker

## Worker Instructions

```sh
temporal server start-dev
```

```sh
python worker.py
```

## Run Instructions

Given `topic` and `articles`

```sh
temporal workflow start \
  --type GenerateTweetWorkflow \
  --task-queue default \
  --workflow-id generate-tweet-workflow-$(date +%s) \
  --input '{"topic":"Artificial Intelligence","articles":["AI breakthrough in natural language processing","Machine learning transforms healthcare diagnostics","Ethical considerations in AI development"]}'
```

## Todo

- Run temporal worker with mock code
- Run a workflow
- Visualise runing a worklfow in Temporal UI
- Store row in DB in python
