# Diamond Characteristics and Prices Data Analysis

## Overview

In this data analytics project, we investigate the effects of various diamond characteristics on their prices. The main focus is on the four Cs of diamonds: carat (weight), cut grade, color grade, and clarity grade. We analyze a dataset containing information about approximately 54,000 round-cut diamonds, including their prices and attributes.

## Dataset Overview

The dataset consists of the following attributes for each diamond:

- Carat: The weight of the diamond.
- Cut: The cut quality of the diamond (ordered categorical).
- Color: The color grade of the diamond (ordered categorical).
- Clarity: The clarity grade of the diamond (ordered categorical).
- Table: The width of the top of the diamond relative to the widest point.
- Depth: The height of the diamond from the culet to the table.
- Dimensions: The dimensions of the diamond.
- Price: The price of the diamond.

Before conducting the analysis, we cleaned the data by removing inconsistent or missing data, ensuring its integrity.

## Distribution of Diamond Prices

The distribution of diamond prices in the dataset shows a large range of values, from approximately $300 to $20,000 at the highest. When plotted on a logarithmic scale, the distribution takes on a multimodal shape, indicating variations in diamond prices.

## Distribution of Carat Weights

Carat weight, representing the size of the diamond, exhibits a distribution with peaks at specific carat sizes. The most common carat sizes are to one decimal place (e.g., 0.3, 0.7, 1.0) or slightly larger.

## Price vs. Diamond Size

When examining the relationship between price and diamond size, taking the cube root of carat weight and plotting it against price on a logarithmic scale results in an approximately linear relationship. However, for carat weights above 1.5, there appears to be a price ceiling, suggesting that some larger diamonds may exceed $20,000.

## Price and Diamond Size by Clarity Grade

An interaction effect is observed between price, diamond size, and the clarity grade. As clarity grade increases from the lowest (I1) to the highest (IF), there are fewer diamonds of size around 1 carat and more diamonds of size around 0.3 carats. Additionally, there is an increase in price as the clarity level rises, with a corresponding change in carat weight.

## Price by Color and Clarity for Selected Carat Weights

Analyzing price by color and clarity for diamonds of around 0.3 carats and 1 carat, we observe an increase in price as clarity level increases. Similarly, within each clarity level, pricing tends to increase with better color grades.

## Price by Cut and Clarity for Selected Carat Weights

Reproducing the previous plots with cut instead of color grade shows that pricing increases with cut quality grade as well. However, the overall effect of cut appears to be smaller compared to color.

## Conclusion

This data analytics project provides valuable insights into the relationship between diamond characteristics and their prices. It demonstrates that carat weight, cut grade, color grade, and clarity grade play significant roles in determining the price of a diamond. The findings can be valuable for consumers, jewelers, and the diamond industry in general to understand the factors influencing diamond prices and make informed decisions in purchasing and selling diamonds.
