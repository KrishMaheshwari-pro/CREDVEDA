import app
from flask import session
from app import api_explainability_latest

with app.app.test_request_context('/api/explainability/EBAY/latest'):
    session['logged_in'] = True
    try:
        result = api_explainability_latest('EBAY')
        print('result type:', type(result))
        if hasattr(result, 'get_data'):
            print(result.get_data(as_text=True))
        else:
            print(result)
    except Exception as e:
        import traceback
        traceback.print_exc()