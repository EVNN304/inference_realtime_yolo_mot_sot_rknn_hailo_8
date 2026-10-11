from additional_scripts.main_src.geometry_and_math.geometry_lib import *
from datetime import datetime
import os
from additional_scripts.main_src.protocols.toolset import *

class Save_detect:
    def __init__(self, q_in:mp.JoinableQueue, size_queue=1, path_to_save="", folder_drone="drone/", folder_bird="bird/", folder_plane="plane/", folder_no_det="no_det/", size_cut_w=800, size_cut_h=800, min_conf=0.9, classes_names={}):
        self.q_in = q_in
        self.size_queue = size_queue
        self.path_to_save = path_to_save
        self.folder_drone = folder_drone
        self.folder_bird = folder_bird
        self.folder_plane = folder_plane
        self.folder_no_det = folder_no_det
        self.size_cut_w, self.size_cut_h = size_cut_w, size_cut_h
        self.min_conf = min_conf
        self.names_files = None
        self.classes_names = classes_names
        self.dict_path_obj = {}

    def set_size_queue(self, size_q):
        self.size_queue = size_q


    def set_names_file(self, name:str):
        self.names_files = name

    def get_names_file(self):
        return self.names_files

    def set_min_conf(self, conf):
        self.min_conf = conf

    def set_size_cut_w(self, w):
        self.size_cut_w = w

    def set_size_cut_h(self, h):
        self.size_cut_h = h




    def set_folder_no_det(self, name_folder):
        self.folder_no_det = name_folder+"/"

    def set_path_save(self, path):
        try:
            self.path_to_save = path+"/"
            os.makedirs(self.path_to_save, exist_ok=True)
            os.makedirs(self.path_to_save + self.folder_no_det, exist_ok=True)
            for i, k in enumerate(self.classes_names.items()):
                os.makedirs(self.path_to_save + self.classes_names[i] + "/", exist_ok=True)
                self.dict_path_obj[i] = self.path_to_save + self.classes_names[i] + "/"

        except Exception as e:
            print(f"Directories have already been created: {e.args}")

    def get_size_queue(self):
        return self.size_queue


    def get_path_save(self):
        return self.path_to_save

    def main_start_process(self):
        process_save = mp.Process(target=self.work_to_save, args=(), daemon=True)
        process_save.start()

    def write_file(self, names, detect_norm, recording_mode="a"):
        with open(names, recording_mode) as file:
            file.write(str(int(detect_norm[0])) + " " + str(detect_norm[1]) + " " + str(detect_norm[2]) + " " + str(detect_norm[3]) + " " + str(detect_norm[4]) + "\n")




    def decode_dets(self, lst_dets):
        lst_decode = []
        for det in lst_dets:
            x, y = det.get_int_center()
            w, h = det.get_wh()
            cls_obj = det.obj_class
            confidence = det.p
            if confidence <= self.min_conf:
                lst_decode.append([cls_obj, x, y, w, h])
        return lst_decode


    def cord_normal(self, lst, w_norm, h_norm, flag_w, x, y):
        if flag_w:
            return [lst[0], round((lst[1] - x)/w_norm, 2), round((lst[2] - y)/h_norm, 2), round(lst[3]/w_norm, 2), round(lst[4]/h_norm, 2)]
        else:
            return [1, (lst[0] - x)/w_norm, (lst[1] - y)/h_norm, lst[2]/w_norm, lst[3]/h_norm]


    def recycling(self, files, lst_dec, lst_dec_i, crd_calc, flag_w):
        if flag_w:
            for j in range(len(lst_dec)):
                if (crd_calc[0]  < lst_dec[j][1] < crd_calc[2] ) and (crd_calc[1] < lst_dec[j][2] < crd_calc[3]) and (lst_dec_i[1::] != lst_dec[j][1::]):
                    det_norm = self.cord_normal(lst_dec[j], self.size_cut_w, self.size_cut_h, flag_w, crd_calc[0], crd_calc[1])
                    self.write_file(files, det_norm,  recording_mode="a")

        else:
            for j in range(len(lst_dec)):
                if (crd_calc[0]  < lst_dec[j][1][1] < crd_calc[2] ) and (crd_calc[1] < lst_dec[j][1][2] < crd_calc[3]) and (lst_dec_i != lst_dec[j][1]):
                    det_norm = self.cord_normal(lst_dec[j][1], self.size_cut_w, self.size_cut_h, flag_w, crd_calc[0], crd_calc[1])
                    self.write_file(files, det_norm, recording_mode="a")





    def work_to_save(self):


        while True:
            if not self.q_in.empty():
                image, lst_dets, flag_w = self.q_in.get()
                if len(image.shape) == 3:
                    H, W, _ = image.shape
                if len(image.shape) < 3:
                    H, W = image.shape

                if flag_w:
                    lst_decode = self.decode_dets(lst_dets)
                    if len(lst_dets) > 1:
                        for i in range(len(lst_decode)):

                            cur_date_1, t_1 = datetime.today(), datetime.now().time()
                            filename = f'{self.names_files}_{self.classes_names[lst_decode[i][0]]}_{str(cur_date_1.day)}_{str(cur_date_1.month)}_{cur_date_1.year}_{"_".join(str(t_1).split(":"))[:-7]}_{str(t_1).split(".")[-1]}'

                            cord_calc_1 = cover_pt_by_area((lst_decode[i][1], lst_decode[i][2]), area_w_h=[self.size_cut_w, self.size_cut_h], limit_box=[0, 0, W, H])
                            cv.imwrite(f'{self.dict_path_obj[lst_decode[i][0]]}{filename}_{str(self.size_cut_w)}x{str(self.size_cut_h)}.jpg', image[cord_calc_1[1]:cord_calc_1[3], cord_calc_1[0]:cord_calc_1[2]])
                            self.write_file(f'{self.dict_path_obj[lst_decode[i][0]]}{filename}_{str(self.size_cut_w)}x{str(self.size_cut_h)}.txt', self.cord_normal(lst_decode[i], self.size_cut_w, self.size_cut_h, flag_w, cord_calc_1[0], cord_calc_1[1]), recording_mode="a")
                            self.recycling(f'{self.dict_path_obj[lst_decode[i][0]]}{filename}_{str(self.size_cut_w)}x{str(self.size_cut_h)}.txt', lst_decode, lst_decode[i], cord_calc_1, flag_w)


                    else:
                        for i in range(len(lst_decode)):


                            cur_date_1, t_1 = datetime.today(), datetime.now().time()
                            print(f"fffff", self.classes_names, lst_decode[i][0])
                            filename = f'{self.names_files}_{self.classes_names[lst_decode[i][0]]}_{str(cur_date_1.day)}_{str(cur_date_1.month)}_{cur_date_1.year}_{"_".join(str(t_1).split(":"))[:-7]}_{str(t_1).split(".")[-1]}'
                            cord_calc_1 = cover_pt_by_area((lst_decode[i][1], lst_decode[i][2]), area_w_h=[self.size_cut_w, self.size_cut_h], limit_box=[0, 0, W, H])
                            print("FUCK_YOU", lst_decode)
                            cv.imwrite(f'{self.dict_path_obj[lst_decode[i][0]]}{filename}_{str(self.size_cut_w)}x{str(self.size_cut_h)}.jpg', image[cord_calc_1[1]:cord_calc_1[3], cord_calc_1[0]:cord_calc_1[2]])
                            self.write_file(f'{self.dict_path_obj[lst_decode[i][0]]}{filename}_{str(self.size_cut_w)}x{str(self.size_cut_h)}.txt', self.cord_normal(lst_decode[i], self.size_cut_w, self.size_cut_h, flag_w, cord_calc_1[0], cord_calc_1[1]), recording_mode="a")
                else:
                    if len(lst_dets) > 1:
                        for i, k in enumerate(lst_dets):
                            cur_date_4, t_4 = datetime.today(), datetime.now().time()
                            filename = f'{self.names_files}_no_det_{str(cur_date_4.day)}_{str(cur_date_4.month)}_{cur_date_4.year}_{"_".join(str(t_4).split(":"))[:-7]}_{str(t_4).split(".")[-1]}'
                            cord_calc_4 = cover_pt_by_area((k[1][0], k[1][1]), area_w_h=[self.size_cut_w, self.size_cut_h], limit_box=[0, 0, W, H])


                            cv.imwrite(f'{self.path_to_save + self.folder_no_det}{filename}_{str(self.size_cut_w)}x{str(self.size_cut_h)}.jpg', image[cord_calc_4[1]:cord_calc_4[3], cord_calc_4[0]:cord_calc_4[2]])
                            self.write_file(f'{self.path_to_save + self.folder_no_det}{filename}_{str(self.size_cut_w)}x{str(self.size_cut_h)}.txt', self.cord_normal(lst_dets[i][1], self.size_cut_w, self.size_cut_h, flag_w, cord_calc_4[0], cord_calc_4[1]), recording_mode="a")
                            self.recycling(f'{self.path_to_save + self.folder_no_det}{filename}_{str(self.size_cut_w)}x{str(self.size_cut_h)}.txt', lst_dets, lst_dets[i][1], cord_calc_4, flag_w)

                    else:
                        for i, k in enumerate(lst_dets):
                            cur_date_4, t_4 = datetime.today(), datetime.now().time()

                            filename = f'{self.names_files}_no_det_{str(cur_date_4.day)}_{str(cur_date_4.month)}_{cur_date_4.year}_{"_".join(str(t_4).split(":"))[:-7]}_{str(t_4).split(".")[-1]}'
                            cord_calc_4 = cover_pt_by_area((k[1][0], k[1][1]), area_w_h=[self.size_cut_w, self.size_cut_h], limit_box=[0, 0, W, H])

                            cv.imwrite(f'{self.path_to_save + self.folder_no_det}{filename}_{str(self.size_cut_w)}x{str(self.size_cut_h)}.jpg', image[cord_calc_4[1]:cord_calc_4[3], cord_calc_4[0]:cord_calc_4[2]])
                            self.write_file(f'{self.path_to_save + self.folder_no_det}{filename}_{str(self.size_cut_w)}x{str(self.size_cut_h)}.txt', self.cord_normal(lst_dets[i][1], self.size_cut_w, self.size_cut_h, flag_w, cord_calc_4[0], cord_calc_4[1]), recording_mode="a")

                self.q_in.task_done()



