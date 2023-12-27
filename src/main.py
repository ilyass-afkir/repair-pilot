"""
This program executes various functions for data preprocessing, data analytics, and clustering based on the given dataframes.

Usage:
    >>> main(product_type, start_date, end_date)

Args:
    product_type (str): Type of the product to be analyzed.
    start_date (str): Start date of the analysis period in the format "dd.mm.yyyy".
    end_date (str): End date of the analysis period in the format "dd.mm.yyyy".

Functions:
    data_preprocessing_main(): Function that preprocesses the raw data.
    data_analytics_main(product_type, start_date, end_date): Function that analyzes the preprocessed data using unsupervised ML annd more!
    
Authors:
    TU Darmstadt - Department of Product Life Cycle Management (PLCM)

Version:
    1.0.0

Date:
    25.04.2023

License:
    MIT

"""

# Import Functions
from backend.data_preprocessing import data_preprocessing_main
from backend.data_analytics import data_analytics_main


def main(product_type, start_date, end_date):
    
    data_preprocessing_main()

    data_analytics_main(product_type, start_date, end_date)


if __name__ == "__main__":
    print("------------------------------------------------------------------------------------")
    print(f"\033[1mAI-based assistace system for product repair and re-design\033[0m")
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print("Welcome to the AI-based assistance system for product repair and re-design.")
    print("This program performs data preprocessing, analytics, and machine learning tasks on a\nspecified product type within a given date range.")
    print("")
    print("Please provide the following details:")
    print("")
    
    """
    # Get user input for parameters
    product_type = input("Enter product type: ")
    start_date = input("Enter start date (dd.mm.yyyy): ")
    end_date = input("Enter end date (dd.mm.yyyy): ")
    """
    # Execute main function with user input
    main("MAK-18", "01.07.2017", "01.07.2021")

    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print(f"\033[1mEnd of Program\033[0m")
    print("------------------------------------------------------------------------------------")
    

  

