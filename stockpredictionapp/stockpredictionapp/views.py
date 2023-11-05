from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
#from .scrape_stock_data import scrape_stock_data

""" 
I want my website to be very simple. 
A textbox which lets the user input a stock code and after submitting, sends a get request to the server. 
The server will attempt to scrape the data. 
If unsuccessful, an error is returned, 
otherwise we will plot the data (stock close value) for everyday for a year (or shorter if the stock has not been up for a year).
"""

def stockInputAPIView(APIView):
    def get(self, request):
        # Retrieve the stock code from the request query parameters
        stock_code = request.query_params.get('stock_code')
        
        if not stock_code:
            return JsonResponse({'error': 'Missing stock_code parameter'}, status=status.HTTP_400_BAD_REQUEST)

        # Call your scraping function to get the Excel file or data
        stock_data = None #scrape_stock_data(stock_code)

        if stock_data:
            # If data is successfully scraped, return it as JSON
            return JsonResponse(stock_data)
        else:
            # If scraping fails, provide an error message
            return JsonResponse({'error': 'Failed to retrieve stock data'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)