import json
from urllib.request import urlopen

def close(message):
    return {
        'sessionAttributes': {},
        'dialogAction': {
            'type': 'Close',
            'fulfillmentState': 'Fulfilled',
            'message': {
                'contentType': 'PlainText',
                'content': message
            }
        }
    }

def getQuote(request):
    slots = request['currentIntent']['slots']
    ticker = slots['ticker']
    price = callQuoteAPI(ticker)
    message = 'The last price (delayed) of ticker {} was {}'.format(ticker, price)
    return close(message)


def callQuoteAPI(ticker):
    response = urlopen('https://api.iextrading.com/1.0/stock/{}/delayed-quote'.format(ticker))
    response = json.load(response)
    return response['delayedPrice']


def lambda_handler(event, context):
    # TODO implement
    intent = event['currentIntent']['name']
    if intent == 'GetQuote':
        return getQuote(event)
    else:
        return 'Sorry'