# Diamonds-analysis
analyzing Diamonds in the market

Effects of Diamond Characteristics on Their Prices
Investigation Overview
In this investigation, I wanted to look at the characteristics of diamonds that could be used to predict their prices. The main focus was on the four Cs of diamonds: carat (weight), cut grade, color grade, and clarity grade.

Dataset Overview
The data consisted of prices and attributes of approximately 54,000 round-cut diamonds. The attributes included the four Cs of diamonds, as well as additional measurements such as table, depth, and dimensions. Thirty-nine data points were removed from the analysis due to inconsistencies or missing information.

# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

%matplotlib inline

# suppress warnings from final output
import warnings
warnings.simplefilter("ignore")
# load in the dataset into a pandas dataframe
diamonds = pd.read_csv('diamonds.csv')
# convert cut, color, and clarity into ordered categorical types
ordinal_var_dict = {'cut': ['Fair','Good','Very Good','Premium','Ideal'],
                    'color': ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
                    'clarity': ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']}

for var in ordinal_var_dict:
    ordered_var = pd.api.types.CategoricalDtype(ordered = True,
                                                categories = ordinal_var_dict[var])
    diamonds[var] = diamonds[var].astype(ordered_var)
# data wrangling, removing diamonds with inconsistent or missing data.
incorrect_depth = (np.abs(2 * diamonds['z'] / (diamonds['x'] + diamonds['y']) - diamonds['depth']/100) > 0.1)
no_size_info = ((diamonds['x'] == 0) & (diamonds['y'] == 0))
diamonds = diamonds.loc[-incorrect_depth & -no_size_info,:]
Distribution of Diamond Prices
Diamond prices in the dataset take on a very large range of values, from about 
19,000 at the highest. Plotted on a logarithmic scale, the distribution of diamond prices takes on a multimodal shape.

log_binsize = 0.025
bins = 10 ** np.arange(2.4, np.log10(diamonds['price'].max())+log_binsize, log_binsize)

plt.figure(figsize=[8, 5])
plt.hist(data = diamonds, x = 'price', bins = bins)
plt.title('Distribution of Diamond Prices')
plt.xscale('log')
plt.xlim([300, 2e4])
plt.xticks([500, 1e3, 2e3, 5e3, 1e4, 2e4], [500, '1k', '2k', '5k', '10k', '20k'])
plt.xlabel('Price ($)')
plt.show()

Distribution of Carat Weights
Carat is a measure of a diamond's weight, where 1 carat is equivalent to 0.2 grams. A large proportion of diamonds take on carat sizes to one decimal place (e.g. 0.3, 0.7, 1.0) or slightly larger, with gradually decreasing frequencies until the next peak is reached.

binsize = 0.02
bins = np.arange(0.2, 2.5 + binsize, binsize)

plt.figure(figsize=[8, 5])
plt.hist(data = diamonds, x = 'carat', bins = bins)
plt.title('Distribution of Carat Weights')
plt.xlim([0.2,2.5])
plt.xlabel('Carat')
plt.gca().set_xticks(np.arange(0.2, 2.5, 0.1), minor = True)
plt.show()

Price vs. Diamond Size
Plotting price on a logarithmic scale and taking the cube root of carat (which is approximately proportional to the dimensions of the diamond) produces an approximately linear relationship. For carat weights above 1.5, there appears to be a price ceiling: based on the trend below 1.5 carats, we might expect some larger diamonds to take prices higher than $20,000.

# compute cube root of carat for modeling and plotting
def cuberoot_trans(x, inverse = False):
    """ quick function for computing cube root and cube operations """
    if not inverse:
        return x ** (1/3)
    else:
        return x ** 3

diamonds['cr_carat'] = diamonds['carat'].apply(cuberoot_trans)
# compute the logarithm of price for other modeling and plotting
def log_trans(x, inverse = False):
    """ quick function for computing log and power operations """
    if not inverse:
        return np.log10(x)
    else:
        return np.power(10, x)

diamonds['log_price'] = diamonds['price'].apply(log_trans)
plt.figure(figsize = [8, 6])
plt.scatter(data = diamonds, x = 'cr_carat', y = 'price', alpha = 1/10)

plt.title('Diamond Prices (log scale) vs. Carat Weight (cube root scale)')

carat_ticks = [0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]
plt.xticks(cuberoot_trans(np.array(carat_ticks)), carat_ticks)
plt.xlim([cuberoot_trans(.18), cuberoot_trans(3.5)])
plt.xlabel('Carat')

plt.yscale('log')
plt.yticks([500, 1e3, 2e3, 5e3, 1e4, 2e4], [500, '1k', '2k', '5k', '10k', '20k'])
plt.ylabel('Price ($)')

plt.show()

Price and Diamond Size by Clarity Grade
There is an interaction effect visible between price, diamond size, and the categorical measures of quality: cut, color, and clarity. This is most evident for the clarity measure. As clarity grade increases from the lowest of I1 to the highest at IF), there are much fewer diamonds of size around 1 carat, and many more diamonds of size around 0.3 carats. At the same time, you can see an increase in one peak from around 
10,000, and a second peak from about 
1,000, whose sizes seem to correspond with the changes in carat weight. This accounts for the base observation that price tends to decrease with increased quality, because the diamonds with higher quality also tend to be smaller in size.

# assist function for plotting 
def catqual_by_pc(var, color):
    fig, ax = plt.subplots(ncols = 2, figsize = [10,6])
    sb.violinplot(data = diamonds, y = var, x = 'log_price', ax = ax[0],
                  color = color)
    ax[0].set_xticks(log_trans(np.array([500, 1e3, 2e3, 5e3, 1e4, 2e4])))
    ax[0].set_xticklabels([500, '1k', '2k', '5k', '10k', '20k'])
    ax[0].set_xlabel("Price ($)")
    ax[0].set_ylabel(var.capitalize())
    sb.violinplot(data = diamonds, y = var, x = 'cr_carat', ax = ax[1],
               color = color)
    ax[1].set_xticks(cuberoot_trans(np.array(carat_ticks)))
    ax[1].set_xticklabels(carat_ticks)
    ax[1].set_xlim(cuberoot_trans(np.array([0.1, 3.5])))
    ax[1].set_xlabel("Carat")
    ax[1].set_yticklabels([""])
    ax[1].set_ylabel("")
    fig.suptitle('Price and Carat Weight by {}'.format(var.capitalize()))
    plt.show()
catqual_by_pc('clarity', sb.color_palette()[0])

Price by Color and Clarity for Selected Carat Weights
We can see how color and clarity grades affect diamond price by plotting the average price on selected carat weights: those around 0.3 carats and those around 1 carat. For both the 0.3-carat and 1.0-carat diamonds, as the clarity level increases, the general height of the points rises, showing an increase in price. Within each level of clarity, we also observe an increase in price based on the color grade (J is lowest, D is highest).

# select diamonds of about 1 carat and 0.3 carat.
diamond_flag = (diamonds['carat'] >= 0.99) & (diamonds['carat'] <= 1.03)
diamonds_1c = diamonds.loc[diamond_flag,:]

diamond_flag = (diamonds['carat'] >= 0.29) & (diamonds['carat'] <= 0.33)
diamonds_sml = diamonds.loc[diamond_flag,:]
fig, ax = plt.subplots(ncols = 2, figsize = [12,6])

sb.pointplot(data = diamonds_sml, x = 'clarity', y = 'price', hue = 'color',
             palette = 'Blues', linestyles = '', dodge = 0.5, ax = ax[0])
ax[0].set_title('0.3-carat diamond prices across color and clarity')
ax[0].set_ylabel('Mean Price ($)')
ax[0].set_yscale('log')
ax[0].set_yticks([400, 600, 1000])
ax[0].set_yticklabels(['400', '600', '1000'])
ax[0].set_yticklabels([],minor = True)

sb.pointplot(data = diamonds_1c, x = 'clarity', y = 'price', hue = 'color',
             palette = 'Blues', linestyles = '', dodge = 0.5, ax = ax[1])
ax[1].set_title('1-carat diamond prices across color and clarity')
ax[1].set_ylabel('Mean Price ($)')
ax[1].set_yscale('log')
ax[1].set_yticks([2e3, 4e3, 6e3, 1e4])
ax[1].set_yticklabels(['2k', '4k', '6k', '10k'])
ax[1].set_yticklabels([],minor = True)

plt.show();

Price by Cut and Clarity for Selected Carat Weights
Reproducing the same plots with cut instead of color grade shows that pricing increases with cut quality grade as well. Compared to color, however, it appears that the overall effect of cut is smaller (from worst to best) compared to color.

fig, ax = plt.subplots(ncols = 2, figsize = [12,6])

sb.pointplot(data = diamonds_sml, x = 'clarity', y = 'price', hue = 'cut',
             palette = 'Greens', linestyles = '', dodge = 0.5, ax = ax[0])
ax[0].set_title('0.3-carat diamond prices across cut and clarity')
ax[0].set_ylabel('Mean Price ($)')
ax[0].set_yscale('log')
ax[0].set_yticks([400, 600, 1000])
ax[0].set_yticklabels(['400', '600', '1000'])
ax[0].set_yticklabels([],minor = True)

sb.pointplot(data = diamonds_1c, x = 'clarity', y = 'price', hue = 'cut',
             palette = 'Greens', linestyles = '', dodge = 0.5, ax = ax[1])
ax[1].set_title('1-carat diamond prices across cut and clarity')
ax[1].set_ylabel('Mean Price ($)')
ax[1].set_yscale('log')
ax[1].set_yticks([2e3, 4e3, 6e3, 1e4])
ax[1].set_yticklabels(['2k', '4k', '6k', '10k'])
ax[1].set_yticklabels([],minor = True)

plt.show();

 
