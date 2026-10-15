# Demo Operations Guide
This is fictional sample documentation for a portfolio demo.
The health endpoint is GET /health. It returns HTTP 200 when the service is ready.
Deployments run after a pull request is merged into main. The pipeline runs unit tests before deployment. If a deployment fails, restore the previous release and verify the health endpoint.
API request limits allow 100 requests per minute per account. Exceeding the limit returns HTTP 429. Retry after the interval in the Retry-After response header.
