'''
VERSION 260627 : 174_POOLS
'''

import json
from web3 import Web3
from math import sqrt, pow

# Connect to Ethereum node
w3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/REPLACE_WITH_YOUR_INFURA_KEY"))

# Pool configurations
# Includes only pools where both tokens exist in the 154 CoinGecko IDs used in the API URL.
# Counts: 89 Uniswap V2 pools and 85 SushiSwap V2 pools. Duplicate symbol-pair names receive an address suffix to avoid overwriting dictionary keys.
UNISWAP_V2_POOLS = {
    'WISE/ETH': {'address': '0x21b8065d10f73EE2e260e5B47D3344d3Ced7596E', 'token0': 'WISE', 'token1': 'ETH'},
    'PEPE/ETH': {'address': '0xA43fe16908251ee70EF74718545e4FE6C5cCEc9f', 'token0': 'PEPE', 'token1': 'ETH'},
    'ETH/USDC': {'address': '0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc', 'token0': 'ETH', 'token1': 'USDC'},
    'ETH/USDT': {'address': '0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852', 'token0': 'ETH', 'token1': 'USDT'},
    'PAXG/ETH': {'address': '0x9C4Fe5FFD9A9fC5678cFBd93Aa2D4FD684b67C4C', 'token0': 'PAXG', 'token1': 'ETH'},
    'SPX/ETH': {'address': '0x52c77b0CB827aFbAD022E6d6CAF2C44452eDbc39', 'token0': 'SPX', 'token1': 'ETH'},
    'ETH/DAI': {'address': '0xA478c2975Ab1Ea89e8196811F51A7B7Ade33eB11', 'token0': 'ETH', 'token1': 'DAI'},
    'WBTC/ETH': {'address': '0xBb2b8038a1640196FbE3e38816F3e67Cba72D940', 'token0': 'WBTC', 'token1': 'ETH'},
    'FLOKI/ETH': {'address': '0xca7c2771D248dCBe09EABE0CE57A62e18dA178c0', 'token0': 'FLOKI', 'token1': 'ETH'},
    'MOG/ETH': {'address': '0xc2eaB7d33d3cB97692eCB231A5D0e4A649Cb539d', 'token0': 'MOG', 'token1': 'ETH'},
    'ELON/ETH': {'address': '0x7B73644935b8e68019aC6356c40661E1bc315860', 'token0': 'ELON', 'token1': 'ETH'},
    'DG/USDC': {'address': '0x873056A02255872514F05249d93228D788Fe4Fb4', 'token0': 'DG', 'token1': 'USDC'},
    'USDC/USDT': {'address': '0x3041CbD36888bECc7bbCBc0045E3B1f144466f5f', 'token0': 'USDC', 'token1': 'USDT'},
    'OPAL/ETH': {'address': '0x5bcebCEe72F13004F1D00D7Da7BF22b082f93f70', 'token0': 'OPAL', 'token1': 'ETH'},
    'HEZ/USDT': {'address': '0xf6C4e4f339912541D3f8ED99Dba64a1372AF5E5B', 'token0': 'HEZ', 'token1': 'USDT'},
    'NEIRO/ETH': {'address': '0xC555D55279023E732CcD32D812114cAF5838fD46', 'token0': 'NEIRO', 'token1': 'ETH'},
    'BEAM/ETH': {'address': '0x180EFC1349A69390aDE25667487a826164C9c6E4', 'token0': 'BEAM', 'token1': 'ETH'},
    'XFI/ETH': {'address': '0xaF996125e98b5804c00FFDB4f7fF386307c99A00', 'token0': 'XFI', 'token1': 'ETH'},
    'BANANA/ETH': {'address': '0x43DE4318b6EB91a7cF37975dBB574396A7b5B5c6', 'token0': 'BANANA', 'token1': 'ETH'},
    'GEKKO/ETH': {'address': '0xDa2d09FBbf8eE4b5051A0e9b562c5FCb4b393b18', 'token0': 'GEKKO', 'token1': 'ETH'},
    'UNI/ETH': {'address': '0xd3d2E2692501A5c9Ca623199D38826e513033a17', 'token0': 'UNI', 'token1': 'ETH'},
    'ASTEROID/ETH': {'address': '0x76A411f14A704099Ba476CE8dFFC288a53295218', 'token0': 'ASTEROID', 'token1': 'ETH'},
    'RAD/USDC': {'address': '0x8C1c499b1796D7F3C2521AC37186B52De024e58c', 'token0': 'RAD', 'token1': 'USDC'},
    'SUPER/ETH': {'address': '0x25647E01Bd0967C1B9599FA3521939871D1d0888', 'token0': 'SUPER', 'token1': 'ETH'},
    'NPC/ETH': {'address': '0x69C7bd26512f52bF6F76faB834140D13Dda673Ca', 'token0': 'NPC', 'token1': 'ETH'},
    'CAW/ETH': {'address': '0x48D20b3e529fB3DD7D91293f80638dF582AB2Daa', 'token0': 'CAW', 'token1': 'ETH'},
    'FLUT/DAI': {'address': '0xE532b04d2F2E921dfEC69E132e9214D2F82DF304', 'token0': 'FLUT', 'token1': 'DAI'},
    'LON/ETH': {'address': '0x7924a818013f39cf800F5589fF1f1f0DEF54F31F', 'token0': 'LON', 'token1': 'ETH'},
    'LFT/ETH': {'address': '0x9c84f58BB51FabD18698efE95F5bAb4F33E96E8f', 'token0': 'LFT', 'token1': 'ETH'},
    'TENSOR/DAI': {'address': '0x0A09cdB390d3B6BA9119843A92c14B2428E101bf', 'token0': 'TENSOR', 'token1': 'DAI'},
    'RAIL/ETH': {'address': '0xac86903cdDA380F20a06Cc8a2DEA7749F1558c68', 'token0': 'RAIL', 'token1': 'ETH'},
    'OLAS/ETH': {'address': '0x09D1d767eDF8Fa23A64C51fa559E0688E526812F', 'token0': 'OLAS', 'token1': 'ETH'},
    'CULT/ETH': {'address': '0x5281E311734869C64ca60eF047fd87759397EFe6', 'token0': 'CULT', 'token1': 'ETH'},
    'DAI/USDC': {'address': '0xAE461cA67B15dc8dc81CE7615e0320dA1A9aB8D5', 'token0': 'DAI', 'token1': 'USDC'},
    'HEZ/DAI': {'address': '0x9bD82673C50acB4A3b883d61e070a3C8D9b08E10', 'token0': 'HEZ', 'token1': 'DAI'},
    'SHIBDOGE/ETH': {'address': '0x3016A43B482d0480460f6625115bd372FE90c6bf', 'token0': 'SHIBDOGE', 'token1': 'ETH'},
    'XCN/ETH': {'address': '0x859f7092f56c43BB48bb46dE7119d9c799716CDF', 'token0': 'XCN', 'token1': 'ETH'},
    'HEZ/ETH': {'address': '0x23d15EDceb5B5B3A23347Fa425846DE80a2E8e5C', 'token0': 'HEZ', 'token1': 'ETH'},
    'MKR/ETH': {'address': '0xC2aDdA861F89bBB333c90c492cB837741916A225', 'token0': 'MKR', 'token1': 'ETH'},
    'JOE/ETH': {'address': '0x704aD8d95C12D7FEA531738faA94402725acB035', 'token0': 'JOE', 'token1': 'ETH'},
    'FOX/ETH': {'address': '0x470e8de2eBaef52014A47Cb5E6aF86884947F08c', 'token0': 'FOX', 'token1': 'ETH'},
    'OCEAN/ETH': {'address': '0x9b7DaD79FC16106b47a3DAb791F389C167e15Eb0', 'token0': 'OCEAN', 'token1': 'ETH'},
    'WOJAK/ETH': {'address': '0xcaA3A16F8440F85303aFaab1992f2b97D12469B1', 'token0': 'WOJAK', 'token1': 'ETH'},
    'TOKEN/ETH': {'address': '0xC7e6B676bfC73Ae40bcC4577F22aab1682C691C6', 'token0': 'TOKEN', 'token1': 'ETH'},
    'LINK/ETH': {'address': '0xa2107FA5B38d9bbd2C461D6EDf11B11A50F6b974', 'token0': 'LINK', 'token1': 'ETH'},
    'CAGA/USDT': {'address': '0x5016cD7B785a773F7f3a3fF4035a1e7A76543946', 'token0': 'CAGA', 'token1': 'USDT'},
    'PEPECOIN/ETH': {'address': '0xDDd23787a6B80A794d952f5fb036D0b31A8E6aff', 'token0': 'PEPECOIN', 'token1': 'ETH'},
    'DUMMY/ETH': {'address': '0xF827FA9B46745A876637406e2A3bf0a2766B89BA', 'token0': 'DUMMY', 'token1': 'ETH'},
    'KISHU/ETH': {'address': '0xF82d8Ec196Fb0D56c6B82a8B1870F09502A49F88', 'token0': 'KISHU', 'token1': 'ETH'},
    'STARL/ETH': {'address': '0xA5e9C917b4B821e4E0A5bbeFce078Ab6540d6B5E', 'token0': 'STARL', 'token1': 'ETH'},
    'WTRX/USDC': {'address': '0x9A84A1852bC7FB608794960960ADb04666A12B41', 'token0': 'WTRX', 'token1': 'USDC'},
    'DSYNC/ETH': {'address': '0x1ffEc7119e315B15852557f654AE0052f76e6ae1', 'token0': 'DSYNC', 'token1': 'ETH'},
    'APU/ETH': {'address': '0x5CEd44F03ff443BBE14d8eA23bc24425FB89E3ED', 'token0': 'APU', 'token1': 'ETH'},
    'MONGOOSE/ETH': {'address': '0x450E653a0A125a1Dc36d3901D3CcE2e2287df0c2', 'token0': 'MONGOOSE', 'token1': 'ETH'},
    'ANDY/ETH': {'address': '0xa1bF0e900FB272089c9fd299EA14BFccb1D1C2c0', 'token0': 'ANDY', 'token1': 'ETH'},
    'MYX/ETH': {'address': '0xE5437565CBa444F33f40215afecC92e38e2d1bA9', 'token0': 'MYX', 'token1': 'ETH'},
    'VOLT/ETH': {'address': '0x96aa22BAEdc5A605357E0b9aE20AB6B10A472E03', 'token0': 'VOLT', 'token1': 'ETH'},
    'PAAL/ETH': {'address': '0x2a6c340bCbb0a79D3deecD3bc5cBc2605ea9259f', 'token0': 'PAAL', 'token1': 'ETH'},
    'STETH/ETH': {'address': '0x4028DAAC072e492d34a3Afdbef0ba7e35D8b55C4', 'token0': 'STETH', 'token1': 'ETH'},
    'FOUR/ETH': {'address': '0xd101821c56B4405Af4A376cBe81FA0dC90207dC2', 'token0': 'FOUR', 'token1': 'ETH'},
    'OIL/USDC': {'address': '0x0E9C8107682AB88604B4FBf847eeeCeaCF38E9E6', 'token0': 'OIL', 'token1': 'USDC'},
    'ETHY/ETH': {'address': '0xBe78353416003aa6e2c38E85249FDEe3Ce8c9B1B', 'token0': 'ETHY', 'token1': 'ETH'},
    'UFO/ETH': {'address': '0x97E1FcB93ae7267dBAFaD23f7b9AFAA08264CFd8', 'token0': 'UFO', 'token1': 'ETH'},
    'LADYS/ETH': {'address': '0xcBE856765eeEc3fDC505dDEbF9dC612Da995e593', 'token0': 'LADYS', 'token1': 'ETH'},
    'GROK/ETH': {'address': '0x69c66BeAfB06674Db41b22CFC50c34A93b8d82a2', 'token0': 'GROK', 'token1': 'ETH'},
    'GME/ETH': {'address': '0x2aEEe741fa1e21120a21E57Db9ee545428E683C9', 'token0': 'GME', 'token1': 'ETH'},
    'SAITABIT/WBTC': {'address': '0x0a8E808f25c8DAFa338AEC17C3736f5b4505A0ed', 'token0': 'SAITABIT', 'token1': 'WBTC'},
    'UNISTAKE/ETH': {'address': '0x78b9524101fb67286338261ddD85E20665e571C1', 'token0': 'UNISTAKE', 'token1': 'ETH'},
    'KEKIUS/ETH': {'address': '0xFFf8D5fFF6Ee3226fa2F5d7D5D8C3Ff785be9C74', 'token0': 'KEKIUS', 'token1': 'ETH'},
    'VLT/ETH': {'address': '0x966053Ca4fca049173eb1F27E4cb168CCb794534', 'token0': 'VLT', 'token1': 'ETH'},
    'AUDIO/ETH': {'address': '0xC730EF0f4973DA9cC0aB8Ab291890D3e77f58F79', 'token0': 'AUDIO', 'token1': 'ETH'},
    'PUPPIES/ETH': {'address': '0x2325e3F261caDb1C30Cebf66c9f95f6fb016c0D4', 'token0': 'PUPPIES', 'token1': 'ETH'},
    'UNIBOT/ETH': {'address': '0x8DbEE21E8586eE356130074aaa789C33159921Ca', 'token0': 'UNIBOT', 'token1': 'ETH'},
    'DPI/ETH': {'address': '0x4d5ef58aAc27d99935E5b6B4A6778ff292059991', 'token0': 'DPI', 'token1': 'ETH'},
    'NATI/ETH': {'address': '0x33A93Bd484053615933502Cb1299f77dE48036C8', 'token0': 'NATI', 'token1': 'ETH'},
    'JESUS/ETH': {'address': '0x8f1B19622a888c53C8eE4F7D7B4Dc8F574ff9068', 'token0': 'JESUS', 'token1': 'ETH'},
    'MASK/USDC': {'address': '0xA40bb1c47F6DD27142a2Bd7C93BFa98db9D1f5c5', 'token0': 'MASK', 'token1': 'USDC'},
    'WOJAK/ETH__0x0F23d49b': {'address': '0x0F23d49bC92Ec52FF591D091b3e16c937034496E', 'token0': 'WOJAK', 'token1': 'ETH'},
    'CHKN/ETH': {'address': '0x5bB8F1Ce603577a4D17cC9D72f6a4C38f3b0b74c', 'token0': 'CHKN', 'token1': 'ETH'},
    'HERA/USDC': {'address': '0x52E2E6b3BA385Ed9690EEF11F72B2fE27ba1F8ca', 'token0': 'HERA', 'token1': 'USDC'},
    'M87/ETH': {'address': '0x2049DF3435bDBB36d22F98FCd2E5027049A1f3Ce', 'token0': 'M87', 'token1': 'ETH'},
    'GOOCH/ETH': {'address': '0xf2cb7243aaf57575E6D3165991327b0410BF91EF', 'token0': 'GOOCH', 'token1': 'ETH'},
    'WOO/ETH': {'address': '0x6AdA49AECCF6E556Bb7a35ef0119Cc8ca795294A', 'token0': 'WOO', 'token1': 'ETH'},
    'FTMX/ETH': {'address': '0x5b2c9EF1bf180a844389C0fD42B15C8281e69052', 'token0': 'FTMX', 'token1': 'ETH'},
    'PEIPEI/ETH': {'address': '0xbF16540c857B4e32cE6C37d2F7725C8eEC869B8b', 'token0': 'PEIPEI', 'token1': 'ETH'},
    'WOLF/ETH': {'address': '0x67324985B5014B36B960273353DEb3D96F2F18C2', 'token0': 'WOLF', 'token1': 'ETH'},
    'AMPL/ETH': {'address': '0xc5be99A02C6857f9Eac67BbCE58DF5572498F40c', 'token0': 'AMPL', 'token1': 'ETH'},
    'QRX/ETH': {'address': '0xF227e97616063A0EA4143744738f9def2aa06743', 'token0': 'QRX', 'token1': 'ETH'},
    'ASTO/USDC': {'address': '0x11181bD3BAF5cE2A478E98361985d42625DE35D1', 'token0': 'ASTO', 'token1': 'USDC'},
}

# Pool configurations
SUSHISWAP_V2_POOLS = {
    'DELTA/ETH': {'address': '0x1498bd576454159bb81b5ce532692a8752d163e8', 'token0': 'DELTA', 'token1': 'ETH'},
    'WBTC/ETH': {'address': '0xceff51756c56ceffca006cd410b03ffc46dd3a58', 'token0': 'WBTC', 'token1': 'ETH'},
    'ILV/ETH': {'address': '0x6a091a3406e0073c3cd6340122143009adac0eda', 'token0': 'ILV', 'token1': 'ETH'},
    'ETH/RAIL': {'address': '0x7cf6a842e72c4bd7bc92e0c17ebdaa95ac1e7825', 'token0': 'ETH', 'token1': 'RAIL'},
    'ETH/ALCX': {'address': '0xc3f279090a47e80990fe3a9c30d24cb117ef91a8', 'token0': 'ETH', 'token1': 'ALCX'},
    'ETH/USDT': {'address': '0x06da0fd433c1a5d7a4faa01111c044910a184553', 'token0': 'ETH', 'token1': 'USDT'},
    'ETH/NFD': {'address': '0xeefa3b448768dd561af4f743c9e925987a1f8d09', 'token0': 'ETH', 'token1': 'NFD'},
    'USDC/QUARTZ': {'address': '0x1e888882d0f291dd88c5605108c72d414f29d460', 'token0': 'USDC', 'token1': 'QUARTZ'},
    'USDC/KIRA': {'address': '0xd3bbb7fb340c0ea4c0c0b60f35a6ac2b116679e3', 'token0': 'USDC', 'token1': 'KIRA'},
    'TOKE/ETH': {'address': '0xd4e7a6e2d03e4e48dfc27dd3f46df1c176647e38', 'token0': 'TOKE', 'token1': 'ETH'},
    'ETH/CIG': {'address': '0x22b15c7ee1186a7c7cffb2d942e20fc228f6e4ed', 'token0': 'ETH', 'token1': 'CIG'},
    'MEEB/ETH': {'address': '0xe339c1d0a744053cbceb0d2dc2d13967c8a69586', 'token0': 'MEEB', 'token1': 'ETH'},
    'USDC/ETH': {'address': '0x397ff1542f962076d0bfe58ea045ffa2d347aca0', 'token0': 'USDC', 'token1': 'ETH'},
    'DAI/RAIL': {'address': '0x7f517657496c09c43cc28d0ca131e35dcd557e02', 'token0': 'DAI', 'token1': 'RAIL'},
    'DAI/ETH': {'address': '0xc3d03e4f041fd4cd388c549ee2a29a9e5075882f', 'token0': 'DAI', 'token1': 'ETH'},
    'CVX/ETH': {'address': '0x05767d9ef41dc40689678ffca0608878fb3de906', 'token0': 'CVX', 'token1': 'ETH'},
    'SUSHI/ETH': {'address': '0x795065dcc9f64b5614c407a6efdc400da6221fb0', 'token0': 'SUSHI', 'token1': 'ETH'},
    'SYN/ETH': {'address': '0x4a86c01d67965f8cb3d0aaa2c655705e64097c31', 'token0': 'SYN', 'token1': 'ETH'},
    'SPELL/ETH': {'address': '0xb5de0c3753b6e1b4dba616db82767f17513e6d4e', 'token0': 'SPELL', 'token1': 'ETH'},
    'ICHI/ETH': {'address': '0x9cd028b1287803250b1e226f0180eb725428d069', 'token0': 'ICHI', 'token1': 'ETH'},
    'XSUSHI/ETH': {'address': '0x36e2fcccc59e5747ff63a03ea2e5c0c2c14911e7', 'token0': 'XSUSHI', 'token1': 'ETH'},
    'LON/USDT': {'address': '0x55d31f68975e446a40a2d02ffa4b0e1bfb233c2f', 'token0': 'LON', 'token1': 'USDT'},
    'STKATOM/ETH': {'address': '0x195b2ed7dfb2bb19c63b8b06677d46934c0c4eea', 'token0': 'STKATOM', 'token1': 'ETH'},
    'IQ/ETH': {'address': '0x9d45081706102e7aaddd0973268457527722e274', 'token0': 'IQ', 'token1': 'ETH'},
    'FTX TOKEN/ETH': {'address': '0x7825de5586e4d2fd04459091bbe783fa243e1bf3', 'token0': 'FTX TOKEN', 'token1': 'ETH'},
    'ETH/SHX': {'address': '0xea59a3da9cad8ece406a353ddede0e7d17604d28', 'token0': 'ETH', 'token1': 'SHX'},
    'YFI/ETH': {'address': '0x088ee5007c98a9677165d78dd2109ae4a3d04d0c', 'token0': 'YFI', 'token1': 'ETH'},
    'ETH/ENS': {'address': '0xa1181481beb2dc5de0daf2c85392d81c704bf75d', 'token0': 'ETH', 'token1': 'ENS'},
    'FRAX/ETH': {'address': '0xec8c342bc3e07f05b9a782bc34e7f04fb9b44502', 'token0': 'FRAX', 'token1': 'ETH'},
    'COMP/ETH': {'address': '0x31503dcb60119a812fee820bb7042752019f2355', 'token0': 'COMP', 'token1': 'ETH'},
    'AAVE/ETH': {'address': '0xd75ea151a61d06868e31f8988d28dfe5e9df57b4', 'token0': 'AAVE', 'token1': 'ETH'},
    'MULTI/ETH': {'address': '0x82917fb0dd65b0e5c85eea66e4f5c1ed484bc629', 'token0': 'MULTI', 'token1': 'ETH'},
    'BTC2X-FLI/WBTC': {'address': '0x164fe0239d703379bddde3c80e4d4800a1cd452b', 'token0': 'BTC2X-FLI', 'token1': 'WBTC'},
    'UMA/ETH': {'address': '0x001b6450083e531a5a7bf310bd2c1af4247e23d4', 'token0': 'UMA', 'token1': 'ETH'},
    'FODL/USDC': {'address': '0xa5c475167f03b1556c054e0da78192cd2779087f', 'token0': 'FODL', 'token1': 'USDC'},
    'RAIL/RENBTC': {'address': '0x519854dd32b4f34f9a407874b0b69218465972d6', 'token0': 'RAIL', 'token1': 'RENBTC'},
    'ETH/AMP': {'address': '0x15e86e6f65ef7ea1dbb72a5e51a07926fb1c82e3', 'token0': 'ETH', 'token1': 'AMP'},
    'BAND/ETH': {'address': '0xa75f7c2f025f470355515482bde9efa8153536a8', 'token0': 'BAND', 'token1': 'ETH'},
    'PENDLE/ETH': {'address': '0x37922c69b08babcceae735a31235c81f1d1e8e43', 'token0': 'PENDLE', 'token1': 'ETH'},
    'ETH/AMPL': {'address': '0xcb2286d9471cc185281c4f763d34a962ed212962', 'token0': 'ETH', 'token1': 'AMPL'},
    'DIA/USDC': {'address': '0xc965bc2cf23d54ccf41a7d3af10572580e9b8078', 'token0': 'DIA', 'token1': 'USDC'},
    'C98/ETH': {'address': '0x557b08a0cab46bfe22471b65522757ea92a9e7a5', 'token0': 'C98', 'token1': 'ETH'},
    'GOG/ETH': {'address': '0x5c596c6a65f628fc1090853d8eb1927651e9d9b2', 'token0': 'GOG', 'token1': 'ETH'},
    'OHM/DAI': {'address': '0x055475920a8c93cffb64d039a8205f7acc7722d3', 'token0': 'OHM', 'token1': 'DAI'},
    'MIC/USDT': {'address': '0xc9cb53b48a2f3a9e75982685644c1870f1405ccb', 'token0': 'MIC', 'token1': 'USDT'},
    'NEAR/ETH': {'address': '0x6469b34a2a4723163c4902dbbdea728d20693c12', 'token0': 'NEAR', 'token1': 'ETH'},
    'EETH/ETH': {'address': '0x6db0fe375ccb8ac3c2a0984d678bcd99981ddcb6', 'token0': 'EETH', 'token1': 'ETH'},
    'WBTC/DIGG': {'address': '0x9a13867048e01c663ce8ce2fe0cdae69ff9f35e3', 'token0': 'WBTC', 'token1': 'DIGG'},
    'DDX/USDC': {'address': '0xc2b0f2a7f736d3b908bdde8608177c8fc28c1690', 'token0': 'DDX', 'token1': 'USDC'},
    'RADAR/ETH': {'address': '0x559ebe4e206e6b4d50e9bd3008cda7ce640c52cb', 'token0': 'RADAR', 'token1': 'ETH'},
    'VUSD/ETH': {'address': '0xb90047676cc13e68632c55cb5b7cbd8a4c5a0a8e', 'token0': 'VUSD', 'token1': 'ETH'},
    'OHM/DAI__0x34d7d7aa': {'address': '0x34d7d7aaf50ad4944b70b320acb24c95fa2def7c', 'token0': 'OHM', 'token1': 'DAI'},
    'BANK/ETH': {'address': '0x938625591adb4e865b882377e2c965f9f9b85e34', 'token0': 'BANK', 'token1': 'ETH'},
    'ETH/IMX': {'address': '0x18cd890f4e23422dc4aa8c2d6e0bd3f3bd8873d8', 'token0': 'ETH', 'token1': 'IMX'},
    'KP3R/ETH': {'address': '0xaf988aff99d3d0cb870812c325c588d8d8cb7de8', 'token0': 'KP3R', 'token1': 'ETH'},
    'OMNI/ETH': {'address': '0x39f49254d6eaf6b2b2549dfec4ed93cf6bae167f', 'token0': 'OMNI', 'token1': 'ETH'},
    'NST/ETH': {'address': '0x75507bab932fc47531bf1654f243c088a3fa1f69', 'token0': 'NST', 'token1': 'ETH'},
    'DOG/ETH': {'address': '0xc96f20099d96b37d7ede66ff9e4de59b9b1065b1', 'token0': 'DOG', 'token1': 'ETH'},
    'SNX/ETH': {'address': '0xa1d7b2d891e3a1f9ef4bbc5be20630c2feb1c470', 'token0': 'SNX', 'token1': 'ETH'},
    'BIFI/ETH': {'address': '0x0bec54c89a7d9f15c4e7faa8d47adedf374462ed', 'token0': 'BIFI', 'token1': 'ETH'},
    'ETH/BICO': {'address': '0x55d8ec728ea72477c6db12ca497a803c8db361e9', 'token0': 'ETH', 'token1': 'BICO'},
    'SFI/ETH': {'address': '0x23a9292830fc80db7f563edb28d2fe6fb47f8624', 'token0': 'SFI', 'token1': 'ETH'},
    'MOONCAT/ETH': {'address': '0x0aa1e808bba7cde5210d13d23ee726de37b8a8bb', 'token0': 'MOONCAT', 'token1': 'ETH'},
    'FODL/ETH': {'address': '0xce7e98d4da6ebda6af474ea618c6b175729cd366', 'token0': 'FODL', 'token1': 'ETH'},
    'NFTX/ETH': {'address': '0x31d64f9403e82243e71c2af9d8f56c7dbe10c178', 'token0': 'NFTX', 'token1': 'ETH'},
    'MANA/ETH': {'address': '0x1bec4db6c3bc499f3dbf289f5499c30d541fec97', 'token0': 'MANA', 'token1': 'ETH'},
    'GCAKE/ETH': {'address': '0x230813772d6d3788dfcd3e7b65fb3dd371b3d8d6', 'token0': 'GCAKE', 'token1': 'ETH'},
    'SUSHI/FRAX': {'address': '0xe06f8d30ac334c857fc8c380c85969c150f38a6a', 'token0': 'SUSHI', 'token1': 'FRAX'},
    'THOR/ETH': {'address': '0x3d3f13f2529ec3c84b2940155effbf9b39a8f3ec', 'token0': 'THOR', 'token1': 'ETH'},
    'LDO/ETH': {'address': '0xc558f600b34a5f69dd2f0d06cb8a88d829b7420a', 'token0': 'LDO', 'token1': 'ETH'},
    'ETH/DOGEGF': {'address': '0xa416df4d96cd547337a3e8893bf3f01c2a2af5c0', 'token0': 'ETH', 'token1': 'DOGEGF'},
    'OHM/LUSD': {'address': '0xfdf12d1f85b5082877a6e070524f50f6c84faa6b', 'token0': 'OHM', 'token1': 'LUSD'},
    'WASABI/ETH': {'address': '0x8f9ef75cd6e610dd8acf8611c344573032fb9c3d', 'token0': 'WASABI', 'token1': 'ETH'},
    'DPI/ETH': {'address': '0x34b13f8cd184f55d0bd4dd1fe6c07d46f245c7ed', 'token0': 'DPI', 'token1': 'ETH'},
    'UNI/ETH': {'address': '0xdafd66636e2561b0284edde37e42d192f2844d40', 'token0': 'UNI', 'token1': 'ETH'},
    'LINK/ETH': {'address': '0xc40d16476380e4037e6b1a2594caf6a6cc8da967', 'token0': 'LINK', 'token1': 'ETH'},
    'ETH/GRT': {'address': '0x7b504a15ef05f4eed1c07208c5815c49022a0c19', 'token0': 'ETH', 'token1': 'GRT'},
    'RARE/ETH': {'address': '0x24e2adf50c49633039757ab08bff489b7c5bd6b2', 'token0': 'RARE', 'token1': 'ETH'},
    'OCEAN/ETH': {'address': '0xee35e548c7457fcdd51ae95ed09108be660ea374', 'token0': 'OCEAN', 'token1': 'ETH'},
    'DAI/WSTETH': {'address': '0xc5578194d457dcce3f272538d1ad52c68d1ce849', 'token0': 'DAI', 'token1': 'WSTETH'},
    'POND/ETH': {'address': '0xccb42aaab2ea1f8012c30418ee3fce550954d827', 'token0': 'POND', 'token1': 'ETH'},
    'NEWO/USDC': {'address': '0xc08ed9a9abeabcc53875787573dc32eee5e43513', 'token0': 'NEWO', 'token1': 'USDC'},
    'ETH/QTO': {'address': '0x2dfe706831a12f0887115f5de1ca72fce4a1204f', 'token0': 'ETH', 'token1': 'QTO'},
    'ETH/WNCG': {'address': '0x877d9c970b8b5501e95967fe845b7293f63e72f7', 'token0': 'ETH', 'token1': 'WNCG'},
    'OHM/ETH': {'address': '0xfffae4a0f4ac251f4705717cd24cadccc9f33e06', 'token0': 'OHM', 'token1': 'ETH'},
}



# ABI definitions
V2_PAIR_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"internalType": "uint112", "name": "_reserve0", "type": "uint112"},
            {"internalType": "uint112", "name": "_reserve1", "type": "uint112"},
            {"internalType": "uint32", "name": "_blockTimestampLast", "type": "uint32"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {"constant": True, "inputs": [], "name": "token0", "outputs": [{"type": "address"}], "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [], "name": "token1", "outputs": [{"type": "address"}], "stateMutability": "view", "type": "function"}
]


ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    }
]


def get_UNISWAP_v2_pool_data(pool_address, pool_name, token0_name, token1_name):
    try:
        pool = w3.eth.contract(address=w3.to_checksum_address(pool_address), abi=V2_PAIR_ABI)
        reserve0, reserve1, timestamp = pool.functions.getReserves().call()

        token0_address = pool.functions.token0().call()
        token1_address = pool.functions.token1().call()

        try:
            token0 = w3.eth.contract(address=token0_address, abi=ERC20_ABI)
            token0_symbol = token0.functions.symbol().call()
            token0_decimals = token0.functions.decimals().call()
        except:
            token0_symbol = token0_name
            token0_decimals = 18

        try:
            token1 = w3.eth.contract(address=token1_address, abi=ERC20_ABI)
            token1_symbol = token1.functions.symbol().call()
            token1_decimals = token1.functions.decimals().call()
        except:
            token1_symbol = token1_name
            token1_decimals = 18

        # Calculate adjusted reserves
        adjusted_reserve0 = reserve0 / (10 ** token0_decimals)
        adjusted_reserve1 = reserve1 / (10 ** token1_decimals)

        # Calculate price
        price = adjusted_reserve1 / adjusted_reserve0 if adjusted_reserve0 > 0 else 0

        return {
            'pool': pool_name,
            'reserves': {
                token0_symbol: str(reserve0),
                token1_symbol: str(reserve1)
            },
            'reserves_adjusted': {
                token0_symbol: adjusted_reserve0,
                token1_symbol: adjusted_reserve1
            },
            'price': price,
            'last_update': timestamp,
            'token0': token0_symbol,
            'token1': token1_symbol,
            'token0_address': token0_address,
            'token1_address': token1_address,
            'token0_decimals': token0_decimals,
            'token1_decimals': token1_decimals,
            'type': 'UNISWAP_V2'
        }
    except Exception as e:
        print(f"Error processing UNISWAP V2 pool {pool_name}: {str(e)}")
        return None



def get_SUSHISWAP_v2_pool_data(pool_address, pool_name, token0_name, token1_name):
    try:
        pool = w3.eth.contract(address=w3.to_checksum_address(pool_address), abi=V2_PAIR_ABI)
        reserve0, reserve1, timestamp = pool.functions.getReserves().call()

        token0_address = pool.functions.token0().call()
        token1_address = pool.functions.token1().call()

        try:
            token0 = w3.eth.contract(address=token0_address, abi=ERC20_ABI)
            token0_symbol = token0.functions.symbol().call()
            token0_decimals = token0.functions.decimals().call()
        except:
            token0_symbol = token0_name
            token0_decimals = 18

        try:
            token1 = w3.eth.contract(address=token1_address, abi=ERC20_ABI)
            token1_symbol = token1.functions.symbol().call()
            token1_decimals = token1.functions.decimals().call()
        except:
            token1_symbol = token1_name
            token1_decimals = 18

        # Calculate adjusted reserves
        adjusted_reserve0 = reserve0 / (10 ** token0_decimals)
        adjusted_reserve1 = reserve1 / (10 ** token1_decimals)

        # Calculate price
        price = adjusted_reserve1 / adjusted_reserve0 if adjusted_reserve0 > 0 else 0

        return {
            'pool': pool_name,
            'reserves': {
                token0_symbol: str(reserve0),
                token1_symbol: str(reserve1)
            },
            'reserves_adjusted': {
                token0_symbol: adjusted_reserve0,
                token1_symbol: adjusted_reserve1
            },
            'price': price,
            'last_update': timestamp,
            'token0': token0_symbol,
            'token1': token1_symbol,
            'token0_address': token0_address,
            'token1_address': token1_address,
            'token0_decimals': token0_decimals,
            'token1_decimals': token1_decimals,
            'type': 'SUSHISWAP_V2'
        }
    except Exception as e:
        print(f"Error processing SUSHISWAP V2 pool {pool_name}: {str(e)}")
        return None




# Collect all pool data
all_pools_data = []

print("=== UNISWAP V2 Pools ===")
for pool_name, config in UNISWAP_V2_POOLS.items():
    pool_data = get_UNISWAP_v2_pool_data(config['address'], pool_name, config['token0'], config['token1'])
    if pool_data:
        all_pools_data.append(pool_data)
        print(f"{pool_name}:")
        print(f"  {pool_data['token0']} reserve: {pool_data['reserves_adjusted'][pool_data['token0']]:.1f} {pool_data['token0']}")
        print(f"  {pool_data['token1']} reserve: {pool_data['reserves_adjusted'][pool_data['token1']]:.1f} {pool_data['token1']}")
        print(f"  Price: {pool_data['price']:.6f} {pool_data['token1']}/{pool_data['token0']}")
        print(f"  Last update: {pool_data['last_update']}\n")

print("=== SUSHISWAP V2 Pools ===")
for pool_name, config in SUSHISWAP_V2_POOLS.items():
    pool_data = get_SUSHISWAP_v2_pool_data(config['address'], pool_name, config['token0'], config['token1'])
    if pool_data:
        all_pools_data.append(pool_data)
        print(f"{pool_name}:")
        print(f"  {pool_data['token0']} reserve: {pool_data['reserves_adjusted'][pool_data['token0']]:.1f} {pool_data['token0']}")
        print(f"  {pool_data['token1']} reserve: {pool_data['reserves_adjusted'][pool_data['token1']]:.1f} {pool_data['token1']}")
        print(f"  Price: {pool_data['price']:.6f} {pool_data['token1']}/{pool_data['token0']}")
        print(f"  Last update: {pool_data['last_update']}\n")

# Save to JSON file
with open('uniswap_v2_and_sushiswap_v2_pools_data.json', 'w') as f:
    json.dump(all_pools_data, f, indent=2)

print("Data saved to uniswap_v2_and_sushiswap_v2_pools_data.json")