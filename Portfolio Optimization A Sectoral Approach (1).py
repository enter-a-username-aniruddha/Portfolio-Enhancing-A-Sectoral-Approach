#!/usr/bin/env python
# coding: utf-8

# ### Importing  and installing the required libraries

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().system('pip install yfinance')
get_ipython().system('pip install PyPortfolioOpt')
get_ipython().system('pip install plotly')


# ### Webscrapping the top 564 companies based on their market size also mapping the companies to their respective sectors.

# In[ ]:


import yfinance as yf
from requests.exceptions import HTTPError
import pandas as pd

# List of ticker symbols
ticker_symbols = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDB",
    "IBN",
    "INFY",
    "SBIN.NS",
    "LICI.NS",
    "BHARTIARTL.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "LT.NS",
    "HCLTECH.NS",
    "BAJFINANCE.NS",
    "SUNPHARMA.NS",
    "ADANIENT.NS",
    "MARUTI.NS",
    "TATAMOTORS.NS",
    "KOTAKBANK.NS",
    "ONGC.NS",
    "AXISBANK.BO",
    "TITAN.NS",
    "NTPC.NS",
    "ADANIGREEN.NS",
    "ULTRACEMCO.NS",
    "ASIANPAINT.NS",
    "ADANIPORTS.NS",
    "WIT",
    "COALINDIA.NS",
    "POWERGRID.NS",
    "BAJAJFINSV.NS",
    "DMART.NS",
    "NESTLEIND.NS",
    "IOC.NS",
    "M&M.NS",
    "BAJAJ-AUTO.NS",
    "DLF.NS",
    "ADANIPOWER.NS",
    "HAL.NS",
    "IRFC.NS",
    "JSWSTEEL.NS",
    "VBL.NS",
    "TATASTEEL.NS",
    "LTIM.NS",
    "SIEMENS.NS",
    "SBILIFE.NS",
    "BEL.NS",
    "GRASIM.NS",
    "ZOMATO.NS",
    "PNB.NS",
    "PIDILITIND.NS",
    "BANKBARODA.NS",
    "TRENT.NS",
    "PFC.NS",
    "BPCL.NS",
    "HINDZINC.NS",
    "TECHM.NS",
    "GODREJCP.NS",
    "IOB.NS",
    "HDFCLIFE.NS",
    "INDIGO.NS",
    "RECLTD.NS",
    "TATAPOWER.NS",
    "ADANIENSOL.NS",
    "AMBUJACEM.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "GAIL.NS",
    "HINDALCO.NS",
    "INDUSINDBK.NS",
    "ABB.NS",
    "ATGL.NS",
    "TATACONSUM.NS",
    "UNIONBANK.NS",
    "EICHERMOT.NS",
    "RDY",
    "LODHA.NS",
    "CANBK.NS",
    "TVSMOTOR.NS",
    "VEDL.NS",
    "IDBI.NS",
    "BAJAJHLDNG.NS",
    "APOLLOHOSP.NS",
    "DIVISLAB.NS",
    "SHREECEM.NS",
    "DABUR.NS",
    "ZYDUSLIFE.NS",
    "CHOLAFIN.NS",
    "NHPC.NS",
    "SHRIRAMFIN.NS",
    "HEROMOTOCO.NS",
    "HAVELLS.NS",
    "TORNTPHARM.NS",
    "MANKIND.NS",
    "IDEA.NS",
    "INDHOTEL.NS",
    "BOSCHLTD.NS",
    "JSWENERGY.NS",
    "MAXHEALTH.NS",
    "UNITDSPR.BO",
    "ICICIGI.NS",
    "YESBANK.NS",
    "MOTHERSON.NS",
    "CUMMINSIND.NS",
    "HINDPETRO.NS",
    "GICRE.NS",
    "IRCTC.NS",
    "ICICIPRULI.NS",
    "LUPIN.NS",
    "SRF.NS",
    "TIINDIA.NS",
    "INDIANB.NS",
    "POLYCAB.NS",
    "SBICARD.NS",
    "NMDC.NS",
    "MARICO.NS",
    "COLPAL.NS",
    "NAUKRI.NS",
    "BERGEPAINT.NS",
    "GODREJPROP.NS",
    "PERSISTENT.NS",
    "OIL.NS",
    "ALKEM.NS",
    "MRF.NS",
    "BANKINDIA.NS",
    "CONCOR.NS",
    "SOLARINDS.NS",
    "AUROPHARMA.NS",
    "ABBOTINDIA.NS",
    "SUZLON.NS",
    "INDUSTOWER.NS",
    "PATANJALI.NS",
    "CENTRALBK.NS",
    "IDFCFIRSTB.NS",
    "LTTS.NS",
    "PIIND.NS",
    "PGHH.NS",
    "TORNTPOWER.NS",
    "MUTHOOTFIN.NS",
    "SAIL.NS",
    "JSL.NS",
    "GMRINFRA.NS",
    "ASTRAL.NS",
    "BHARATFORG.NS",
    "TATACOMM.NS",
    "MMYT",
    "MPHASIS.NS",
    "ASHOKLEY.NS",
    "ACC.NS",
    "PHOENIXLTD.NS",
    "OBEROIRLTY.NS",
    "SUPREMEIND.NS",
    "PRESTIGE.NS",
    "TATAELXSI.NS",
    "ABCAPITAL.NS",
    "SJVN.NS",
    "LINDEINDIA.NS",
    "AWL.NS",
    "NIACL.NS",
    "UBL.NS",
    "PSB.NS",
    "POLICYBZR.NS",
    "SCHAEFFLER.NS",
    "BALKRISIND.NS",
    "MAHABANK.NS",
    "NYKAA.NS",
    "KPITTECH.NS",
    "PETRONET.NS",
    "THERMAX.NS",
    "COFORGE.NS",
    "DIXON.NS",
    "PAGEIND.NS",
    "AUBANK.NS",
    "DALBHARAT.NS",
    "APLAPOLLO.NS",
    "HUDCO.NS",
    "GUJGASLTD.NS",
    "FLUOROCHEM.NS",
    "FEDERALBNK.NS",
    "UPL.NS",
    "CRISIL.NS",
    "UNOMINDA.NS",
    "VOLTAS.NS",
    "AIAENG.NS",
    "LICHSGFIN.NS",
    "NLCINDIA.NS",
    "3MINDIA.NS",
    "DELHIVERY.NS",
    "FORTIS.NS",
    "APOLLOTYRE.NS",
    "HONAUT.NS",
    "JKCEMENT.NS",
    "BANDHANBNK.NS",
    "MFSL.NS",
    "BIOCON.NS",
    "JUBLFOOD.NS",
    "COROMANDEL.NS",
    "BDL.NS",
    "DEEPAKNTR.NS",
    "ESCORTS.NS",
    "GLAND.NS",
    "IPCALAB.NS",
    "IGL.NS",
    "KIOCL.NS",
    "ITI.NS",
    "SYNGENE.NS",
    "BSE.NS",
    "NATIONALUM.NS",
    "EXIDEIND.NS",
    "BAYERCROP.NS",
    "ZFCVINDIA.NS",
    "JBMA.NS",
    "NBCC.NS",
    "BLUESTARCO.NS",
    "GLENMARK.NS",
    "ENDURANCE.NS",
    "APARINDS.NS",
    "HINDCOPPER.NS",
    "TATACHEM.NS",
    "KANSAINER.NS",
    "EIHOTEL.NS",
    "AARTIIND.NS",
    "PAYTM.NS",
    "SUNTV.NS",
    "HATSUN.NS",
    "MANYAVAR.NS",
    "GRINDWELL.NS",
    "WNS",
    "IIFL.NS",
    "CYIENT.NS",
    "SKFINDIA.NS",
    "TRIDENT.NS",
    "RADICO.NS",
    "RATNAMANI.NS",
    "LAURUSLABS.NS",
    "APLLTD.NS",
    "SANOFI.NS",
    "TIMKEN.NS",
    "PEL.NS",
    "RELAXO.NS",
    "CARBORUNIV.NS",
    "RAMCOCEM.NS",
    "LALPATHLAB.NS",
    "EMAMILTD.NS",
    "ELGIEQUIP.NS",
    "DEVYANI.NS",
    "KAJARIACER.NS",
    "INOXWIND.NS",
    "MAHINDCIE.NS",
    "CDSL.NS",
    "RNW",
    "SUMICHEM.NS",
    "CROMPTON.NS",
    "MCX.NS",
    "IDFC.NS",
    "ATUL.NS",
    "BATAINDIA.NS",
    "NATCOPHARM.NS",
    "RITES.NS",
    "PPLPHARMA.NS",
    "TTML.NS",
    "IIFLWAM.NS",
    "CHALET.NS",
    "OLECTRA.NS",
    "KEC.NS",
    "INDIAMART.NS",
    "ZEEL.NS",
    "RBLBANK.NS",
    "WHIRLPOOL.NS",
    "REDINGTON.NS",
    "CENTURYPLY.NS",
    "WELSPUNIND.NS",
    "BLS.NS",
    "MGL.NS",
    "DCMSHRIRAM.NS",
    "FINCABLES.NS",
    "CHAMBLFERT.NS",
    "BLUEDART.NS",
    "KARURVYSYA.NS",
    "PVR.NS",
    "HBLPOWER.NS",
    "BASF.NS",
    "ALOKINDS.NS",
    "SWSOLAR.NS",
    "CHENNPETRO.NS",
    "KSB.NS",
    "VGUARD.NS",
    "FINPIPE.NS",
    "JSLHISAR.NS",
    "TANLA.NS",
    "SCHNEIDER.NS",
    "RKFORGE.NS",
    "FINEORG.NS",
    "GMDCLTD.NS",
    "AMBER.NS",
    "ACE.NS",
    "ASAHIINDIA.NS",
    "IEX.NS",
    "TEJASNET.NS",
    "BEML.NS",
    "MMTC.NS",
    "RAYMOND.NS",
    "ENGINERSIN.NS",
    "HAPPSTMNDS.NS",
    "NETWORK18.NS",
    "ZENSARTECH.NS",
    "ECLERX.NS",
    "BBTC.NS",
    "GRAPHITE.NS",
    "CEATLTD.NS",
    "BAJAJELEC.NS",
    "SFL.NS",
    "ANANTRAJ.NS",
    "AAVAS.NS",
    "PCBL.NS",
    "AETHER.NS",
    "SPARC.NS",
    "KFINTECH.NS",
    "EIDPARRY.NS",
    "GRANULES.NS",
    "ELECTCAST.NS",
    "INGERRAND.NS",
    "IBULHSGFIN.NS",
    "AMARAJABAT.NS",
    "GPIL.NS",
    "ZYDUSWELL.NS",
    "CUB.NS",
    "RAJESHEXPO.NS",
    "RPOWER.NS",
    "INFIBEAM.NS",
    "MAPMYINDIA.NS",
    "CERA.NS",
    "SAFARI.NS",
    "MAHLIFE.NS",
    "PRAJIND.NS",
    "JUBLPHARMA.NS",
    "NEULANDLAB.NS",
    "CRAFTSMAN.NS",
    "MASTEK.NS",
    "RELINFRA.NS",
    "SYRMA.NS",
    "GALAXYSURF.NS",
    "KALPATPOWR.NS",
    "KTKBANK.NS",
    "EASEMYTRIP.NS",
    "MIDHANI.NS",
    "RCF.NS",
    "POWERMECH.NS",
    "ESABINDIA.NS",
    "QUESS.NS",
    "FORCEMOT.NS",
    "VARROC.NS",
    "VIPIND.NS",
    "RELIGARE.NS",
    "CAMPUS.NS",
    "KIRLFER.NS",
    "ARVIND.NS",
    "JSWHL.NS",
    "SOUTHBANK.NS",
    "SHRIPISTON.NS",
    "VOLTAMP.NS",
    "STAR.NS",
    "FDC.NS",
    "KNRCON.NS",
    "VAIBHAVGBL.NS",
    "TEXRAIL.NS",
    "BORORENEW.NS",
    "JINDWORLD.NS",
    "EDELWEISS.NS",
    "GOKEX.NS",
    "RAIN.NS",
    "MARKSANS.NS",
    "SUNTECK.NS",
    "PDSL.NS",
    "VESUVIUS.NS",
    "ITDC.NS",
    "PAISALO.NS",
    "SPANDANA.NS",
    "HCC.NS",
    "SURYAROSNI.NS",
    "HEG.NS",
    "GREENLAM.NS",
    "ETHOSLTD.NS",
    "TCI.NS",
    "MSTCLTD.NS",
    "PRINCEPIPE.NS",
    "TATACOFFEE.NS",
    "SYMPHONY.NS",
    "JPASSOCIAT.NS",
    "ASTRAMICRO.NS",
    "CSBBANK.NS",
    "GUJALKALI.NS",
    "PURVA.NS",
    "STLTECH.NS",
    "PTC.NS",
    "NAZARA.NS",
    "MOIL.NS",
    "IFBIND.NS",
    "RTNPOWER.NS",
    "SUPRAJIT.NS",
    "WONDERLA.NS",
    "GATEWAY.NS",
    "ORIENTCEM.NS",
    "GABRIEL.NS",
    "VRLLOG.NS",
    "NFL.NS",
    "ASHOKA.NS",
    "TIMETECHNO.NS",
    "AARTIDRUGS.NS",
    "SUNDARMHLD.NS",
    "GULFOILLUB.NS",
    "HEIDELBERG.NS",
    "EMUDHRA.NS",
    "KKCL.NS",
    "WABAG.NS",
    "BBOX.NS",
    "TINPLATE.NS",
    "GREENPANEL.NS",
    "ORIENTELEC.NS",
    "DISHTV.NS",
    "WSTCSTPAPR.NS",
    "HATHWAY.NS",
    "HGS.NS",
    "SUDARSCHEM.NS",
    "SPICEJET.BO",
    "IMAGICAA.NS",
    "UNITECH.NS",
    "SUBROS.NS",
    "JISLDVREQS.NS",
    "KSCL.NS",
    "DELTACORP.NS",
    "SKIPPER.NS",
    "NUCLEUS.NS",
    "JTEKTINDIA.NS",
    "TATASTLLP.NS",
    "SEQUENT.NS",
    "PFOCUS.NS",
    "LUXIND.NS",
    "CONFIPET.NS",
    "KSL.NS",
    "HIKAL.NS",
    "INDOCO.NS",
    "CARERATING.NS",
    "IMFA.NS",
    "PRAKASH.NS",
    "AVALON.NS",
    "SANDHAR.NS",
    "SHALBY.NS",
    "PARAGMILK.NS",
    "ASHIANA.NS",
    "FILATEX.NS",
    "TIDEWATER.NS",
    "JCHAC.NS",
    "CIGNITITEC.NS",
    "NILKAMAL.NS",
    "SANGHIIND.NS",
    "GREENPLY.NS",
    "QUICKHEAL.NS",
    "ACCELYA.NS",
    "AUTOAXLES.NS",
    "REPCOHOME.NS",
    "GTLINFRA.NS",
    "MTNL.NS",
    "APOLLOPIPE.NS",
    "POLYPLEX.NS",
    "VSTTILLERS.NS",
    "SOMANYCERA.NS",
    "PFS.NS",
    "MAHLOG.NS",
    "DOLLAR.NS",
    "ANUP.NS",
    "RPGLIFE.NS",
    "INDIAGLYCO.NS",
    "HUHTAMAKI.NS",
    "VADILALIND.NS",
    "KCP.NS",
    "JAGRAN.NS",
    "SASKEN.NS",
    "ALEMBICLTD.NS",
    "HINDOILEXP.NS",
    "MPSLTD.NS",
    "JINDALPOLY.NS",
    "WENDT.NS",
    "SMLISUZU.NS",
    "HONDAPOWER.NS",
    "KINGFA.NS",
    "TCNSBRANDS.NS",
    "FOSECOIND.NS",
    "BARBEQUE.NS",
    "MMFL.NS",
    "GTPL.NS",
    "SIYSIL.NS",
    "RUPA.NS",
    "LUMAXIND.NS",
    "CAPACITE.NS",
    "VIDHIING.NS",
    "HIL.NS",
    "CANTABIL.NS",
    "OMAXE.NS",
    "FMGOETZE.NS",
    "HSIL.NS",
    "GEPIL.NS",
    "SPIC.NS",
    "MANGLMCEM.NS",
    "SIFY",
    "GEOJITFSL.NS",
    "TNPL.NS",
    "SHANKARA.NS",
    "GALLISPAT.NS",
    "NELCO.NS",
    "ASTEC.NS",
    "GATI.NS",
    "KITEX.NS",
    "INDNIPPON.NS",
    "NAVKARCORP.NS",
    "DHAMPURSUG.NS",
    "RANEHOLDIN.NS",
    "IGARASHI.NS",
    "HIMATSEIDE.NS",
    "WHEELS.NS",
    "IGPL.NS",
    "INEOSSTYRO.NS",
    "NACLIND.NS",
    "MANGCHEFER.NS",
    "KUANTUM.NS",
    "COSMOFIRST.NS",
    "BALAJITELE.NS",
    "JAYBARMARU.NS",
    "MONTECARLO.NS",
    "APTECHT.NS",
    "IMPAL.NS",
    "INDORAMA.NS",
    "KOKUYOCMLN.NS",
    "COFFEEDAY.NS",
    "BLISSGVS.NS",
    "STERTOOLS.NS",
    "RICOAUTO.NS",
    "OAL.NS",
    "HESTERBIO.NS",
    "SPENCERS.NS",
    "ORIENTPPR.NS",
    "EXCELINDUS.NS",
    "BODALCHEM.NS",
    "KELLTONTEC.NS",
    "NECLIFE.NS",
    "HMVL.NS",
    "3IINFOLTD.NS",
    "YTRA",
    "ASIANTILES.NS",
    "RADIOCITY.NS",
    "APEX.NS",
    "MIRZAINT.NS",
    "EBIXFOREX.NS",
    "RCOM.NS",
    "RBL.NS",
    "NBIFIN.NS",
    "NXTDIGITAL.NS",
    "ZEELEARN.NS",
    "MEP.NS",
    "MLKFOOD.BO",
    "FRETAIL.NS",
    "SREINFRA.NS",
    "JUMPNET.NS",
    "MODAIRY.BO",
    "FSC.NS"

]

# Counters
total_companies = len(ticker_symbols)
discarded_companies = 0
available_companies = 0

# Set start and end dates
start_date = '2014-01-01'
end_date = '2024-01-01'

# Create an empty DataFrame to store data for companies with sector and industry
all_stock_data = pd.DataFrame()

# Loop through each ticker symbol
for ticker_symbol in ticker_symbols:
    try:
        # Create a Ticker object
        ticker = yf.Ticker(ticker_symbol)

        # Get the info dictionary containing various information including sector and industry
        info = ticker.info

        # Get sector and industry information
        sector = info.get('sector')
        industry = info.get('industry')

        # Check if both sector and industry are not None
        if sector is not None and industry is not None:
            # Download historical data for the current ticker symbol
            stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

            # Add stock information as columns to the DataFrame
            stock_data['Company Name'] = info['longName']
            stock_data['Sector'] = sector
            stock_data['Industry'] = industry

            # Append the stock data to the DataFrame for all stocks
            all_stock_data = pd.concat([all_stock_data, stock_data])

            available_companies += 1
        else:
            discarded_companies += 1
    except HTTPError as e:
        if e.response.status_code == 404:
            print(f"Error 404: Ticker {ticker_symbol} not found. Discarding...")
            discarded_companies += 1

# Print summary
print("Summary:")
print("Total companies input:", total_companies)
print("Number of companies discarded:", discarded_companies)
print("Number of companies available with sector and industry:", available_companies)

# Display the DataFrame
print("\nStock Data:")
print(all_stock_data)


# ### Creating a Dataframe for the scraped data 

# In[ ]:


df = pd.DataFrame(all_stock_data)
df


# ### Count of trading days for each company.

# In[ ]:


company_counts=df['company_counts'] = df['company'].value_counts()
company_counts


# ### Keeping only those companies who have 2465 trading days.

# In[ ]:


num_companies_with_2465 = (company_counts == 2465).sum()
num_companies_with_2465


# ### We are left with 388 companies on which further analysis has been carried out.
# 
# ### Creating a Dataframe for the 388 companies.
# 

# In[ ]:


selected_companies = []
for company, days_count in df.groupby('company')['days_count'].first().items():
    if days_count == 2465:
        selected_companies.append(company)
df1 = df[df['company'].isin(selected_companies)]
df1
df1.drop(columns=['company_counts'], inplace=True)


# ### Computing Previous Close Price and Daily Returns

# In[ ]:


List_of_companies = list(set(df1['company'].unique()))
All_df_list = []
for comp in List_of_companies:
#     print(comp)
    Individual_comp = df1.loc[df1['company'] == comp]
    Individual_comp = Individual_comp.sort_values('Date')
    Individual_comp['prev close'] = Individual_comp['Adj Close'].shift(1)
    Individual_comp['Returns'] = (Individual_comp['Adj Close'] - Individual_comp['prev close'])/Individual_comp['prev close']
    All_df_list.append(Individual_comp)
final_df = pd.concat(All_df_list)
final_df


# ### Creating a Dataframe that only has Company Name and Daily Returns.

# In[ ]:


df2=final_df.pivot(columns="company",values="Returns")
df2


# ### Creating a Dataframe that only has Company Name and Adj Close price.

# In[ ]:


data2=final_df.pivot(columns="company",values="Adj Close")
data2


# ### Creating a Dataframe that comprises Adj Clsoe price for each sector.

# In[ ]:


sector_adj_close = df1.pivot_table(index='Date',columns= 'Sector', values='Adj Close')
sector_adj_close


# ### Building an Efficient Frontier object for Mean-Variance portfolio for the companies.

# In[ ]:


from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

# Get unique sectors and create a dictionary to store sector-wise company names
sectors = df1["Sector"].unique()
sector_companies = {}
for sector in sectors:
    sector_data = df1[df1["Sector"] == sector]
    companies = sector_data["company"].unique().tolist()
    sector_companies[sector] = companies

# Define sector mappings based on the sector-wise company names dictionary
sector_mapper = {company: sector for sector, companies in sector_companies.items() for company in companies}
# Define sector constraints based on the sector-wise company names dictionary
sector_lower = {sector: 0 for sector in sectors}
sector_upper = {sector: 0.1 for sector in sectors}

# Calculate expected returns and the covariance matrix of the portfolio
mu = expected_returns.mean_historical_return(data2)
S = risk_models.sample_cov(data2)

# Create the Efficient Frontier Object
ef = EfficientFrontier(mu, S, solver='ECOS')

# Add sector constraints to the Efficient Frontier
ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

# Optimize for the maximum Sharpe ratio
weights = ef.max_sharpe()

# Clean the raw weights and print them
cleaned_weights = ef.clean_weights()
print(cleaned_weights)


# ### Visualising the weights.

# In[ ]:


import plotly.express as px

# Filter out zero-weighted assets
non_zero_weights = {asset: weight for asset, weight in cleaned_weights.items() if weight != 0}

# Extract labels and values from non-zero weights
labels = list(non_zero_weights.keys())
values = list(non_zero_weights.values())

# Create a pie chart using Plotly
fig = px.pie(names=labels, values=values, title='Non-zero Weight Portfolio Composition')
fig.show()


# ### Building an Efficient Conditional Value at Risk Portfolio for the companies.

# In[ ]:


from pypfopt import EfficientCVaR

# Calculate expected returns and the covariance matrix of the portfolio
mu = expected_returns.mean_historical_return(data2)
S = risk_models.sample_cov(data2)

# Create the Efficient Frontier Object with CVaR
ef = EfficientCVaR(mu, S)

# Add sector constraints to the Efficient Frontier
ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

# Optimize for the minimum CVaR
weights = ef.min_cvar()

# Clean the raw weights and print them
cleaned_weightss = ef.clean_weights()
print(cleaned_weightss)

# Calculate and print the portfolio performance
ef.portfolio_performance(verbose=True)


# ### Visualising the weights.

# In[ ]:


# Filter out zero-weighted assets
non_zero_weightss = {asset: weight for asset, weight in cleaned_weightss.items() if weight != 0}

# Extract labels and values from non-zero weights
labels = list(non_zero_weightss.keys())
values = list(non_zero_weightss.values())

# Create a pie chart using Plotly
fig = px.pie(names=labels, values=values, title='Non-zero Weight Portfolio Composition')
fig.show()


# ### Building an Efficient Frontier object for Mean-Variance portfolio for the sectors.

# In[ ]:


from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

# Get unique sectors and create a dictionary to store sector-wise company names
sectors = df1["Sector"].unique()
sector_companies = {}
for sector in sectors:
    sector_data = df1[df1["Sector"] == sector]
    companies = sector_data["company"].unique().tolist()
    sector_companies[sector] = companies

# Define sector mappings based on the sector-wise company names dictionary
sector_mapper = {company: sector for sector, companies in sector_companies.items() for company in companies}

# Define sector constraints based on the sector-wise company names dictionary
sector_lower = {sector: 0 for sector in sectors}
sector_upper = {sector: 0.1 for sector in sectors}

# Calculate expected returns and the covariance matrix of the portfolio
mu = expected_returns.mean_historical_return(sector_adj_close)
S = risk_models.sample_cov(sector_adj_close)

# Create the Efficient Frontier Object
ef = EfficientFrontier(mu, S, solver='ECOS')

# Add sector constraints to the Efficient Frontier
ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

# Optimize for the maximum Sharpe ratio
weights = ef.max_sharpe()

# Clean the raw weights and print them
cleaned_weights = ef.clean_weights()
print(cleaned_weights)

# Calculate and print the portfolio performance
ef.portfolio_performance(verbose=True)


# ### Visualising the weights for the sectors.

# In[ ]:


# Filter out zero-weighted assets
non_zero_weights = {asset: weight for asset, weight in cleaned_weights.items() if weight != 0}

# Extract labels and values from non-zero weights
labels = list(non_zero_weights.keys())
values = list(non_zero_weights.values())

# Create a pie chart using Plotly
fig = px.pie(names=labels, values=values, title='Non-zero Weight Portfolio Composition')
fig.show()


# ### Building an Efficient Conditional Value at Risk Portfolio for the sectors.

# In[ ]:


# Calculate expected returns and the covariance matrix of the portfolio
mu = expected_returns.mean_historical_return(sector_adj_close)
S = risk_models.sample_cov(sector_adj_close)

# Create the Efficient Frontier Object with CVaR
ef = EfficientCVaR(mu, S, beta= 0.70)

# Add sector constraints to the Efficient Frontier
ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

# Optimize for the minimum CVaR
weights = ef.min_cvar()

# Clean the raw weights and print them
cleaned_weightss = ef.clean_weights()
print(cleaned_weightss)

# Calculate and print the portfolio performance
ef.portfolio_performance(verbose=True)


# ### Visualising the weights for the sectors.

# In[ ]:


# Filter out zero-weighted assets
non_zero_weightss = {asset: weight for asset, weight in cleaned_weightss.items() if weight != 0}

# Extract labels and values from non-zero weights
labels = list(non_zero_weightss.keys())
values = list(non_zero_weightss.values())

# Create a pie chart using Plotly
fig = px.pie(names=labels, values=values, title='Non-zero Weight Portfolio Composition')
fig.show()


# ### MonteCarlo Simulation.

# In[ ]:


import matplotlib.pyplot as plt
import scipy.optimize as sco
table=data2
table


# ### Computing Maximum Sharpe Ratio Allocation and Minimum Volatility Portfolio Allocation for a combination of 20000 portfolios with Risk free rate of 10%.

# In[ ]:


for c in table.columns.values:
    plt.plot(table.index, table[c], lw=3, alpha=0.8, label=c)
plt.legend(loc='upper left', fontsize=9)
plt.ylabel('Price')

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

# Input data preparation
returns = table.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 20000
risk_free_rate = 0.1

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=table.columns, columns=['Allocation'])
    max_sharpe_allocation['Allocation'] = [round(i * 100, 2) for i in max_sharpe_allocation['Allocation']]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=table.columns, columns=['Allocation'])
    min_vol_allocation['Allocation'] = [round(i * 100, 2) for i in min_vol_allocation['Allocation']]
    min_vol_allocation = min_vol_allocation.T

    print("-" * 80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp, 2))
    print("Annualised Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)
    print("-" * 80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min, 2))
    print("Annualised Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)

    plt.figure(figsize=(10, 7))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('Annualised Volatility')
    plt.ylabel('Annualised Returns')
    plt.legend(labelspacing=0.8)
    plt.show()

# Call the function to display simulated efficient frontier
display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)


# ### Discrete Allocation

# In[ ]:


# DISCRETE ALLOCATION OF EFFICIENT FRONTIER
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import pandas as pd

# Function to convert INR to USD
def inr_to_usd(amount_inr):
    # Assuming 1 USD = 75 INR
    return amount_inr / 75.0

portfolio_amount_inr = float(input("Enter the amount you want to invest (INR): "))

# Convert INR to USD
portfolio_amount_usd = inr_to_usd(portfolio_amount_inr)

if portfolio_amount_inr != '':
    # Get discrete allocation of each share per stock
    latest_prices = get_latest_prices(data2)
    weights = cleaned_weights
    discrete_allocation = DiscreteAllocation(weights, latest_prices, total_portfolio_value=int(portfolio_amount_usd))
    allocation, leftover = discrete_allocation.lp_portfolio()

    discrete_allocation_list = []

    for symbol in allocation:
        discrete_allocation_list.append(allocation.get(symbol))

    portfolio_df = pd.DataFrame(columns=['Ticker', 'Number of stocks to buy'])
    portfolio_df['Ticker'] = allocation
    portfolio_df['Number of stocks to buy'] = discrete_allocation_list

    # Convert leftover USD to INR
    leftover_inr = leftover * 75

    print('Number of stocks to buy with the amount of ₨', portfolio_amount_inr)
    print(portfolio_df)
    print('Funds remaining with you will be: ₨', int(leftover_inr))

