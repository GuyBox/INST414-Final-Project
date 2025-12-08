import pandas as pd
import re
from pathlib import Path

def cleanText( text ):
    text = re.sub( r"[^a-zA-Z\s]", "", str( text ) )
    text = text.lower()
    text = " ".join( text.split() )
    return text

def processEbayReviews():
    df = pd.read_csv( "data/raw/ebay_reviews.csv" )
    
    df["text"] = df["review title"].fillna( "" ) + " " + df["review content"].fillna( "" )
    df["cleanedText"] = df["text"].apply( cleanText )
    df["rating"] = df["rating"]
    
    dfClean = df[["cleanedText", "rating"]].dropna()
    
    dfClean.to_csv( "data/processed/ebay_cleaned.csv", index = False )
    print( f"Processed {len( dfClean )} eBay reviews" )
    
    return dfClean

def processAmazonReviews():
    df = pd.read_csv( "data/raw/AmazonReviews.csv" )
    
    df["combinedText"] = df["Summary"].fillna( "" ) + " " + df["Text"].fillna( "" )
    df["cleanedText"] = df["combinedText"].apply( cleanText )
    df["rating"] = df["Score"]
    
    dfClean = df[["cleanedText", "rating"]].dropna()
    
    dfClean.to_csv( "data/processed/amazon_cleaned.csv", index = False )
    print( f"Processed {len( dfClean )} Amazon reviews" )
    
    return dfClean

if __name__ == "__main__":
    ebayDf = processEbayReviews()
    amazonDf = processAmazonReviews()
    
    combinedDf = pd.concat( [ebayDf, amazonDf], ignore_index = True )
    combinedDf.to_csv( "data/processed/combined_cleaned.csv", index = False )
    print( f"Total combined reviews: {len( combinedDf )}" )