import torch
from torch import nn
from torch import optim
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import imageio
from pylab import *
gray();

device = 'cuda'

mit_logo = array(imageio.imread('1280px-MIT_logo.png', as_gray=True))

# frames = imageio.mimread('cctv_gray.gif')
# frames = array(frames)
# frames.shape
# frames = frames.reshape(frames.shape[0],-1)

h, w = mit_logo.shape
W = torch.Tensor((rand(h,w)>0.95).astype('float')).to(device)
imshow(mit_logo)
imshow(W)
imshow(mit_logo*W)
# imageio.imwrite('mit_logo_sampled.png', mit_logo*W.cpu().numpy())
# h,w = frames.shape
# avg = frames.mean(0)
# avg = avg.reshape(134,240)
# imshow(avg)
# %%

r = min(h,w)
L = nn.Parameter(torch.randn(h, r).to(device))
Rt = nn.Parameter(torch.randn(r, w).to(device))

optimizer = optim.Adam([L, Rt], lr=0.05)
X = torch.Tensor(mit_logo).to(device)
# X = torch.randn(h, w).to(device)
# W[:] = 1
# imshow(X)
# with torch.no_grad():
#     Z = L @ Rt
#     imshow(Z)
# %%
gif_images = []
interval = 50
reg_lamb = 50
t = tqdm(range(1000))
for i in t:
    Z = L @ Rt
    rec_loss = torch.norm((Z - X)*W)**2
    reg_loss = reg_lamb * (torch.norm(L)**2 + torch.norm(Rt)**2)
    loss = rec_loss + reg_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        if i % interval==0:
            gif_images.append(((Z.detach().cpu().numpy())))

    t.set_postfix(reg_loss=float(reg_loss), rec=float(rec_loss))

with torch.no_grad():
    Z = L @ Rt
    print(linalg.matrix_rank(Z))
imshow(Z)
# %%
imageio.mimwrite('mit_logo_100.gif', gif_images);

# %%

def complete(X, W, k, lamb, lr, steps=1000, disp=False):
    h, w = X.shape
    L = nn.Parameter(torch.randn(h, k).to(device))
    Rt = nn.Parameter(torch.randn(k, w).to(device))
    optimizer = optim.Adam([L, Rt], lr=0.05)
    if not disp:
        t = range(steps)
    else:
        t = tqdm(range(steps))
    for i in t:
        Z = L @ Rt
        rec_loss = torch.norm((Z - X)*W)**2
        reg_loss = lamb * (torch.norm(L)**2 + torch.norm(Rt)**2)
        loss = rec_loss + reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if disp:
            t.set_postfix(reg_loss=float(reg_loss), rec=float(rec_loss))

    with torch.no_grad():
        Z = L @ Rt
        return Z, linalg.matrix_rank(Z)


# %%
ranks = []
lambs = []
for i,lamb in enumerate(logspace(-5,4,num=10)):
    h, w = 100, 100
    k = 100
    lr = 0.001
    # lamb = 100
    X = torch.randn(h,w).to(device)
    W = torch.Tensor((rand(h,w)>0.5).astype('float')).to(device)
    Z, r = complete(X, W, k, float(lamb), lr, steps=5000)
    ranks.append(r)
    print(r)
    lambs.append(lamb)

rcParams['font.size']=15
figure(figsize=(12,9))
xlabel("$\lambda$ (log-scale)")
ylabel("Rank of reconstructed matrix")
semilogx(lambs, ranks)
show()
# %%
# Z = Z.reshape(57, 134, 240, 1)
# imageio.mimwrite('recons8.gif', Z.cpu().numpy())
#
# A = arange(100).reshape(10,10)
# inds = array([0,3,6])
# inds = tile(inds, (len(inds),1))
# A[inds.T, inds]

# %%
k = 300
mu = 100
l = 1000
rho = 1
eps=1e-5
limit = 500

# %%
def complete2(X, W, k, mu, l, rho, eps=1e-6, limit=100):
    with torch.no_grad():
        U = torch.randn(X.shape[0], k).to(device)
        V = torch.randn(X.shape[1], k).to(device)
        Z = U @ V.transpose(0,1)
        Y = torch.zeros(*X.shape).to(device)
        Ik = torch.eye(k).to(device)
        converged = False
        converged_2 = False
        i = 0
        j = 0
        while not converged:
            j=0
            while not converged_2:
                U = (rho*Z + Y) @ V @ torch.inverse(rho*  V.transpose(0,1) @ V + l*Ik)
                V = (rho*Z + Y).transpose(0,1) @ U @ torch.inverse(rho*U.transpose(0,1) @ U + l*Ik)
                temp = U@V.transpose(0,1) - 1/rho * Y
                Z_ = W * (1/(2+rho)*(2*X + rho*temp)) + (1-W)*temp
                converged_2 = torch.norm(Z-Z_)/torch.norm(Z) < eps or j>limit
                print((torch.norm(Z-Z_)/torch.norm(Z)).item())
                Z = Z_[:]
                j+=1
            temp = (Z - U @ V.transpose(0,1))
            print(torch.norm(temp))
            Y = Y + rho*temp
            rho = min(rho*mu, 1e20)
            converged = torch.norm(temp)<eps or i>limit
            i+=1
imshow(U @ V.transpose(0,1))
imshow(Z)

# %%
gif = imageio.mimread('mit_logo_2.gif')
gif = gif[:75]
imageio.mimwrite('mit_logo_2_short.mp4', gif)

# %%

# with open('netflix-data/combined_data_1.txt') as f:
#     txt = f.read()
#
# lines = txt.split("\n")
# user_id_to_idx = {}
# idx = 0
# t = tqdm(lines)
# for line in t:
#     if len(line)==0:
#         continue
#     if line[-1] != ":":
#         user_id, rating, date = line.split(",")
#         user_id = int(user_id)
#         if user_id_to_idx.get(user_id, -1) == -1:
#             user_id_to_idx[user_id] = idx
#             idx+=1
#
# torch.save(user_id_to_idx, "netflix-data/user_id_to_idx.dict.pkl")
# %%

def read_file(filename):

    user_id_to_idx = torch.load("netflix-data/user_id_to_idx.dict.pkl")
    with open(filename) as f:
        txt = f.read()

    lines = txt.split("\n")
    movies = []
    users = []
    ratings = []
    t = tqdm(lines)
    for line in t:
        if line[-1] == ":":
            movie = int(line[:-1])-1
        else:
            user_id, rating, date = line.split(",")
            movies.append(movie)
            users.append(user_id_to_idx[int(user_id)])
            ratings.append(int(rating))

    i = torch.LongTensor([movies, users])
    v = torch.FloatTensor(ratings)
    return torch.sparse.FloatTensor(i,v,torch.Size([movie+1,len(user_id_to_idx)+1]))

# %%

sp = read_file("netflix-data/combined_data_1.txt")
sp.size()
