
import requests
import json
from datetime import datetime

# API endpoint

url="https://api.coingecko.com/api/v3/simple/price?ids=wise-token11,ethereum,pepe,usd-coin,tether,pax-gold,spx6900,dai,wrapped-bitcoin,floki,mog-coin,dogelon-mars,decentral-games,opal-3,hermez-network-token,neiro,beam-2,crossfi-2,banana-gun,gekko,uniswap,asteroid,radicle,superfarm,non-playable-coin,a-hunters-dream,flute,tokenlon,lifeform,tensor,railgun,autonolas,cult-dao,shibadoge,chain-2,maker,joe,shapeshift-fox-token,ocean-protocol,wojak,tokenfi,chainlink,crypto-asset-governance-alliance,pepecoin-2,dummy,kishu-inu,starlink,wrapped-tron,destra-network,apu-apustaja,mongoose,andy-the-wisguy,myx-finance,volt-inu-2,paal-ai,staked-ether,4,oil,ethy-ai,ufo-gaming,milady-meme-coin,grok-2,game-stop,saitabit,unistake,kekius-maximus,bankroll-vault,audius,i-love-puppies,unibot,defipulse-index,illuminaticoin,jesus-coin,mask-network,chickencoin,hera-finance,messier,gooch,woo-network,ftmtoken,peipeicoin-vip,landwolf,ampleforth,quiverx,altered-state-token,delta-financial,illuvium,alchemix,feisty-doge-nft,sandclock,kira-3,tokemak,cig,meeb-vault-nftx,convex-finance,sushi,synapse-2,spell-token,ichi-farm,xsushi,stkatom,everipedia,ftx-token,stronghold-token,yearn-finance,ethereum-name-service,frax,compound-governance-token,aave,multichain,btc-2x-flexible-leverage-index,uma,fodl-finance,renbtc,amp-token,band-protocol,pendle,dia-data,coin98,guild-of-guardians,olympus,magic-internet-cash,near,ether-fi-staked-eth,digg,derivadao,dappradar,vusd,bankless-dao,immutable-x,keep3rv1,omni-network,ninja-squad,the-doge-nft,havven,bifi,biconomy,saffron-finance,mooncat-vault-nftx,nftx,decentraland,pancake-games,thorswap,lido-dao,dogegf,liquity-usd,wasabi-cheese,the-graph,superrare,wrapped-steth,marlin,new-order,qtoken,wrapped-ncg&vs_currencies=usd"

try:
    # Fetch data from the API
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    
    # current_date = datetime.now().strftime('%Y-%m-%d')
    specified_date = "2026-06-27"

    # Create the data structure to save
    data_to_save = {
        "date": specified_date,
        "prices": response.json()
    }

    # Print the prices
    print(f"Token prices for {specified_date}:")
    for token, price_info in data_to_save["prices"].items():
        print(f"{token.upper()}: ${price_info['usd']}")

    # Save to JSON file
    filename = f"token_prices_{specified_date}.json"
    with open(filename, 'w') as f:
        json.dump(data_to_save, f, indent=4)

    print(f"\nData saved to {filename}")

except requests.exceptions.RequestException as e:
    print(f"Error fetching data from API: {e}")
except Exception as e:
    print(f"An error occurred: {e}")