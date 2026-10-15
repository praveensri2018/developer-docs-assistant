# Demo Order API
This is fictional sample documentation for a portfolio demo.
Create an order using POST /api/orders. The request requires customer_id and items. Each item contains product_id and quantity. Quantity must be a positive integer. A successful request returns HTTP 201 and an order_id.
Authentication uses an Authorization header with a Bearer token. Tokens expire after 60 minutes. Expired tokens return HTTP 401.
