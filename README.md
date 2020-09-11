# air-route
New Version Revising A Mistake In Counting Degrees

Here are some statistics that we have discovered in the USAir97 Dataset which contains 332 nodes(airport) and 2126 edges(flight schedules):

1. avg degree:12.81(edges_num*2/node_num）
2. top10 biggest nodes by degree:[('118', 139), ('261', 118), ('255', 101), ('182', 94), ('152', 94), ('230', 87), ('166', 85), ('67', 78), ('112', 70), ('201', 68)]
   note that the index of the node(e.g. '118') starts from '1'
3. if deliberately delete zero points in the list "degree distribution", then we get Power_Law_Omit_Zero.png, with coeff r=-1.1258094768710132  coeff b=4.394471543376857

4. avg path length:2.7381247042550867(assume adjacent distances are all 1)
   avg path length:1849.7179(with distance amplified 9000 times using 'miles' as the measurement; note that the east-west direct distance in the US is about 2500 miles)
   diameter:6(assume adjacent distances are all 1)---the farthest two are node 9(1 is beginning) and 332，which belongs to Alaska and an western island near Philippines！！！
   diameter:7867.5990(with distance amplified 9000 times)---the farthest two are node 68(1 is beginning) and 330, which belongs to 帕劳(在西太平洋，当时由美国托管) and 缅因州（美国东北角）

5. global clustering coeff:0.6252172491625031. Currently there is not a distribution graph about (local) clustering coeff!!!

6. max_coreness:26
   top35 biggest nodes by coreness:['67', '112', '118', '201', '248', '166', '94', '109', '131', '147', '150', '152', '172', '176', '177', '182', '219', '230', '232', '255', '258', '261', '293', '311', '146', '159', '162', '167', '174', '179', '217', '292', '299', '301', '310']

7. assortativity is -0.20787623081200443, indicating big node connects small node
