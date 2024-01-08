from preprocessing.preprocessing import preprocessing
from features.features import feature_engineering
from analysis.descriptive import descriptive
from analysis.diagnostic import diagnostic
from analysis.predictive import predictive


def main(typ, start_date, end_date):
    
    preprocessing()
    feature_engineering()
    descriptive(typ, start_date, end_date)
    diagnostic(typ, start_date, end_date)
    predictive(typ, start_date, end_date)


if __name__ == "__main__":
    
    #product_type = input("Enter product type: ")
    #start_date = input("Enter start date (dd.mm.yyyy): ")
    #end_date = input("Enter end date (dd.mm.yyyy): ")

    main("MAK-18", "01.07.2017", "01.07.2021")
