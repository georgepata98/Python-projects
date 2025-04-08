# --- Declararea variabilelor ---
distance=[]
energy=[]
fepe=[]
ufepe=[]
sort_distance1=[]
sort_energy1=[]
sort_fepe1=[]
sort_ufepe1=[]
sort_distance2=[]
sort_energy2=[]
sort_fepe2=[]
sort_ufepe2=[]
dist=[]
card_energy=[]
length=0
min=0
card_distance=0
auxx=0
index=0


# --- Citirea datelor din fisierul rezultate.csv ---
f = open("rezultate_fepe.csv", "r")
for x in f:
    values = x.strip().split(",")  # strip() removes any leading/trailing whitespace, including \n at the end of each line; split(",") splits the string using comma as delimiter; x is the current line as a string (including \n)
    distance.append(int(values[0]))
    energy.append(int(values[1]))
    fepe.append(float(values[2]))
    ufepe.append(float(values[3]))
    # print(distance[-1], energy[-1], fepe[-1], ufepe[-1])  # -1 reprezinta ultima valoare adaugata
    length += 1
f.close()
print("\nNumarul total de FEPE este: ", length)


# --- Gasirea distantei minime si a nr. de distante diferite ---
for i in range(length):  # i de la 0 la length-1
    for j in range(length):
        if distance[i] <= distance[j]:
            min += 1
        if i==0 and j==0:
            dist.append(distance[i])
            card_distance += 1
        for k in range(len(dist)):
            if distance[j] != dist[k]:
                auxx += 1
        if auxx == len(dist):
            dist.append(distance[j])
            card_distance += 1
        auxx = 0
    if min == length:
        sort_distance1.append(distance[i])
        sort_energy1.append(energy[i])
        sort_fepe1.append(fepe[i])
        sort_ufepe1.append(ufepe[i])
    min = 0
print("Numarul de distante diferite este:", card_distance)


for i in range(card_distance):
    card_energy.append(0)  # initializam cu 0 pe toate card_energy[i]


# --- Prima sortare dupa cresterea distantei ---
while True:
    aux = len(sort_distance1)
    if aux == length:
        break
    for i in range(length):
        for j in range(length):
            if distance[i] <= distance[j]:
                min += 1
        if min == length-aux:
            sort_distance1.append(distance[i])
            sort_energy1.append(energy[i])
            sort_fepe1.append(fepe[i])
            sort_ufepe1.append(ufepe[i])
        min = 0


# --- Verificare prima sortare ---
verificare_sortare1 = False
if verificare_sortare1 == True:
    for i in range(length):
        print(sort_distance1[i], sort_energy1[i], sort_fepe1[i], sort_ufepe1[i])


# --- Calcularea numarului de energii diferite pentru fiecare distanta ---
for i in range(length):
    if i > 0:
        if sort_distance1[i] != sort_distance1[i-1]:
            for j in range(length):
                if sort_distance1[i-1] == sort_distance1[j]:
                    if sort_energy1[i-1] != sort_energy1[j]:
                        card_energy[index] += 1
            card_energy[index] += 1
            index += 1
    if i == length-1:
        for j in range(length):
            if sort_distance1[i] == sort_distance1[j]:
                if sort_energy1[i] != sort_energy1[j]:
                    card_energy[index] += 1
        card_energy[index] += 1
        index += 1


# --- Sortarea finala dupa cresterea energiei ---
card_energy_len = len(card_energy)  # length=21 pentru distantele 0...20
auxx=0
min=0
index=0
while True:
    if index >= card_energy_len:
        break
    if index > 0:
        auxx += card_energy[index-1]
    ene_per_dist = card_energy[index]
    sum = auxx + ene_per_dist
    aux_per_dist = 0
    while True:
        if aux_per_dist == ene_per_dist:
            break
        for i in range(auxx, sum):
            for j in range(auxx, sum):
                if sort_distance1[i] == sort_distance1[j]:
                    if sort_energy1[i] <= sort_energy1[j]:
                        min += 1
            diff = ene_per_dist - aux_per_dist
            if min == diff:
                sort_distance2.append(sort_distance1[i])
                sort_energy2.append(sort_energy1[i])
                sort_fepe2.append(sort_fepe1[i])
                sort_ufepe2.append(sort_ufepe1[i])
                aux_per_dist += 1
            min = 0
    index += 1


# --- Umplerea fisierului de output ---
f = open("rezultate_fepe_sortate.csv", "w")
for i in range(length):
    f.write(str(sort_distance2[i]) + "," + str(sort_energy2[i]) + "," + str(sort_fepe2[i]) + "," + str(sort_ufepe2[i]) + "\n")
f.close()
print("\nRezultatele sortate sunt in fisierul rezultate_fepe_sortate.csv")