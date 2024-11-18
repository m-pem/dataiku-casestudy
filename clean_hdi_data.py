hdi_1995 = pd.read_csv('raw_data/hdi_1995.csv')

hdi_1995 = hdi_1995.drop(columns=['Unnamed: 0', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6'])

# I now want to update the country column to remove the " [+]" in the string at the end of the value Samoa [+]
hdi_1995['country'] = hdi_1995['country'].str.replace(r'\s*\[\+\]$', '', regex=True)

#I will now update the rank column to remove the 'º' value at the end of the string
hdi_1995['rank'] = hdi_1995['rank'].str.replace('º', '')

# Now I will implement some catogires using the HDI value indicators are compiled into a single number between 0 and 1.0, with 1.0 being the highest possible human development. HDI is divided into four tiers: very high human development (0.8-1.0), high human development (0.7-0.79), medium human development (0.55-.70), and low human development (below 0.55).
hdi_1995['hdi_category'] = pd.cut(hdi_1995['HDI'], bins=[0, 0.55, 0.7, 0.8, 1.0], labels=['low', 'medium', 'high', 'very_high'])

# save out updated data
hdi_1995.to_csv('raw_data/hdi_1995_cleaned.csv', index=False, header=True)