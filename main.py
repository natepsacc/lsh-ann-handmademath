from mlx_embeddings import load, generate
import mlx.core as mx
import itertools
import random
import math
import time

model, tokenizer = load("sentence-transformers/all-MiniLM-L6-v2")

EMBEDDINGS = {}
EMB_NORMAL = {}
RANDOM_VECTS = [] #hyperplanes


def dotProduct(a, b):
    s = 0
    for i in range(len(a[0])):
        s += (a[0][i] * b[0][i]).item()
    return s

def magnitude(v):
    s = 0

    for i in v:
        sq =i*i
        if type(sq) == float:
            s+= sq
        else:
            s+=sq.item()
    
    resp = math.sqrt(s)
    return resp
    

def normalize(v):
    print(v)
    if type(v[0]) == float:
        v_magnitude = magnitude(v)
    else:
        v_magnitude = magnitude(v[0])
    resp = v

    if type(v[0]) == float:
        toIter = v
    else:
        toIter = v[0]
    for i,x in enumerate(toIter):
        if type(resp[0]) == float:
            resp[i] = x/v_magnitude
        else:
            resp[0][i] = x/v_magnitude
    
    return resp

def createEmbeddings(arr):
    for s in arr:
        EMBEDDINGS[s] = generate(model, tokenizer, s)
        EMB_NORMAL[s] = normalize(EMBEDDINGS[s].text_embeds)



def createRandomVectors(n_planes, n_dim):
    resp = []
    for _ in range(n_planes):
        rv = []
        for _ in range(n_dim):
            rv.append(random.gauss(0,1))
        resp.append(normalize(rv))
    return resp

def createVectorHash(v, rvs):
    bitS = []
    for rv in rvs:
        sign =0 # 
        if type(v[0]) == float:
            n_dims = len(v)
        else:
            n_dims = len(v[0])
        for i in range(n_dims):
            if type(v[0]) == float:
                index_val = v[i]
            else:
                index_val = v[0][i]
            print("indexval:",index_val)
            sign += index_val * rv[i] #sign
        bitS.append(1 if sign >0 else 0)
    return tuple(bitS)




def sim(stringA, stringB):
    if stringA in EMBEDDINGS:
        resA = EMBEDDINGS[stringA]
    else:
        resA = generate(model, tokenizer, stringA)
        EMBEDDINGS[stringA] = resA
    if stringB in EMBEDDINGS:
        resB = EMBEDDINGS[stringB]

    else:
        resB = generate(model, tokenizer, stringB)
        EMBEDDINGS[stringB] = resB



    resA_mag = magnitude(resA.text_embeds[0])
    resB_mag = magnitude(resB.text_embeds[0])
    cos_sim = dotProduct(resA.text_embeds, resB.text_embeds) / (resA_mag * resB_mag)
    return cos_sim

strings = ["i like playing golf", "i rode a bike on jupiter", "You cant buy uranium here", 
    "That guy plays football","you live on earth", "pluto isnt a planet", 
    "soccer is a sport", "polo is played on horses", "gold is a metal", "she won silver", "she rode her new horse"]
    # "The afternoon sun stretched across the hills while people wandered slowly through the park, talking about trivial things and enjoying the quiet rhythm of the day.",
    # "In the distance a storm was forming, but the air was still warm and calm, giving the strange feeling that something unusual might happen before nightfall.",
    # "Most people ignored the small details around them, but occasionally someone would notice the subtle patterns that seemed to repeat everywhere.",
    # "The road continued far beyond the town, winding past empty fields and silent buildings that hinted at stories long forgotten.",
    # "Some ideas begin as jokes and eventually turn into serious conversations that change how people think about ordinary things.",
    # "The sky faded from orange to deep blue as the evening settled quietly over the landscape."]

createEmbeddings(strings) # no return as we reuse global var # 
n_dims = len(next(iter(EMBEDDINGS.values())).text_embeds[0])

rvs = createRandomVectors(2, n_dims) #hyperplanes, these will bin our embedded vectors becuase we will know on which side of a given hyperplane or random vector the hash of the embed sits.

v_hashes = [] # normalized embeddings hashes
for v in EMB_NORMAL.values():
    print(v)
    v_hashes.append(createVectorHash(v, rvs))


def search(query):
    embedded_q = normalize(generate(model, tokenizer, query).text_embeds)
    q_vec_hash = createVectorHash(embedded_q, rvs)

    potential_high_sim_vecs = [v for v, h in zip(EMB_NORMAL, v_hashes) if h == q_vec_hash]
    print(potential_high_sim_vecs) 
    simi = 0
    most_sim = None
    for v in potential_high_sim_vecs:
        similiarity = sim(query, v)
        if similiarity > simi:
            simi = similiarity
            most_sim =v
    return most_sim

print(search( "I knew a guy who was from mars"))

# combs = itertools.combinations(strings, 2)
 
# pairsSims = {}
# for c in combs:
#     print(c)
#     simi = sim(c[0], c[1])
#     pairsSims[str(c)] = simi

# sortedResp = dict(sorted(pairsSims.items(), key=lambda item: item[1]))

# for i,x in enumerate(sortedResp):
#     print(x)
