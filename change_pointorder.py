import json
import numpy as np
import matplotlib.pyplot as plt

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def area(vlak, m_points):
    x = []
    y = []
    for p in vlak:
        # print(p)
        x.append(m_points[p][0])
        y.append(m_points[p][1])
    x = np.array(x)
    y = np.array(y)
    area = PolyArea(x, y)
    return area

def get_biggest(vlakken, m_points):
    big = 0
    for i, v in enumerate(vlakken):
        A = area(v, m_points)
        if A > big and x_variation(v, m_points):
            big = A
            index = i
    return index

def change_front(index, vlakken):
    n_front = vlakken[index]
    if (index % 2) == 0: 
        n_back = vlakken[index+1]
    else:
        n_back = vlakken[index-1]
    
    return n_front, n_back

def check_smallest_y(points):
        if points[0][1]<points[1][1]:
            return  points[0]
        else :
            return  points[1]
        
def check_biggest_y(points):
        if points[0][1]>points[1][1]:
            return  points[0]
        else :
            return  points[1]
        
def sort(A):
    return A[np.argsort(A[:, 0])]            
        
def x_variation(vlak, m_points):
    x=[]
    for i in vlak:
        x.append(m_points[i][0])
    x=sorted(x)
    diff=min(abs(x[0]-x[1]), abs(x[2]-x[3]))
    # print("the x values: -> ", diff)
    if diff < 600:
        return True
    else: 
        return False 
        
def main(verbose = False):
    final_json = []
    data_root = "data/synthetic_data/" 

    with open(data_root + "anno.json", 'r') as f:
        data = json.load(f)
    print(f"Length before processing: {len(data)}")
    
    for num, test_dict in enumerate(data):
        m_points = []
        m_points.append(test_dict["projection"][-1]) # center as first point
        m_points.extend(test_dict["projection"][:8])

        if m_points == []:
            Final_list_order=[]
            Final_list_order = np.array(Final_list_order)
        else: 
        # idee oppervlak:
            front = [1,2,3,4]
            back = [5,6,7,8]
            left = [1,4,8,5] # eerste en laatste 
            right = [2,3, 7, 6] # 2 middelste 
            top = [1,2,6,5] # eerste 2 
            bottem = [3,4,8,7] # laatste 2 
            vlakken = [front, back, left, right, top, bottem]

            index = get_biggest(vlakken, m_points) # m_points are projection coordinates
            # index of largest polygon
            n_front, n_back = change_front(index, vlakken)

            if verbose:
                print(index)
                print(n_front)
                print(n_back)
            Real_front = []
            for i in n_front:
                Real_front.append(m_points[i])
            Real_back = []
            for i in n_back:
                Real_back.append(m_points[i])  
            
            Real_front = sort(np.array(Real_front))
            Real_back = sort(np.array(Real_back)) 
            # print(Real_front, Real_back)
                
            Final_list_order=np.zeros_like(m_points)
            
            Final_list_order[8] = m_points[0]
            # onderste links
            Final_list_order[0] = check_smallest_y(Real_back[0:2])
            Final_list_order[1] = check_smallest_y(Real_front[0:2])

            # Boven links
            Final_list_order[2] = check_biggest_y(Real_back[0:2])
            Final_list_order[3] = check_biggest_y(Real_front[0:2])

            # onder rechts
            Final_list_order[4] = check_smallest_y(Real_back[2:4])
            Final_list_order[5] = check_smallest_y(Real_front[2:4])
            # boven rechts
            Final_list_order[6] = check_biggest_y(Real_back[2:4])
            Final_list_order[7] = check_biggest_y(Real_front[2:4])

            Final_list_order = Final_list_order.astype(int)

        final_json.append(
            {
                "image": test_dict["image"], 
                "whd": test_dict["whd"], 
                "projection": Final_list_order.tolist(),
                "world": test_dict["world"]
            })
        
        print(num)
        # debug purpes
        if verbose:
            
            m_img = plt.imread(data_root + "/train/" + str(test_dict["image"]))
            
            # # invert y for plot use
            # h, w, c =m_img.shape
            # for i in range(len(Final_list_order)):
            #     Final_list_order[i][1]=h-Final_list_order[i][1]
            
            for i, p in enumerate(Final_list_order[1:]):
                plt.subplot(2, 5, i+1)
                plt.imshow(m_img)
                plt.plot(p[0], p[1],'ro',markersize=1)
                plt.title(f"{i}")
            plt.show()
    
    print(f"length after processing: {len(final_json)}")

    with open(data_root + "ordered_anno.json", 'w') as f:
        json.dump(final_json, f)

if __name__=="__main__":
    main()

    