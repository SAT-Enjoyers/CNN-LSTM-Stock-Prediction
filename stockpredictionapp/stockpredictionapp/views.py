from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import functionality

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
        stockTag = request.query_params.get('stock_code')
        
        if not stockTag:
            return JsonResponse({'error': 'Missing stock_code parameter'}, status=status.HTTP_400_BAD_REQUEST)

        # Call your scraping function to get the Excel file or data
        stockData = functionality.get_one(1, stockTag)

        if stockData:
            # If data is successfully scraped, return it as JSON
            return JsonResponse(stockData)
        else:
            # If scraping fails, provide an error message
            return JsonResponse({'error': 'Failed to retrieve stock data'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)