import glob
import json
import os
import shutil


def voc_ap(rec, prec):
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap


def file_lines_to_list(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def eval(gt_dict, pred_dict, iou_thresh):
    tmp_files_path = "tmp_files"
    if not os.path.exists(tmp_files_path):
        os.makedirs(tmp_files_path)

    ground_truth_files_list = glob.glob(gt_dict + '/*.txt')
    ground_truth_files_list.sort()
    gt_counter = 0

    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = file_lines_to_list(txt_file)
        bounding_boxes = []
        for line in lines_list:
            class_name, left, top, right, bottom = line.split()
            bbox = left + " " + top + " " + right + " " + bottom
            bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
            gt_counter += 1
        with open(tmp_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    predicted_files_list = glob.glob(pred_dict + '/*.txt')
    predicted_files_list.sort()

    bounding_boxes = []
    for txt_file in predicted_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines = file_lines_to_list(txt_file)
        for line in lines:
            tmp_class_name, confidence, left, top, right, bottom = line.split()
            bbox = left + " " + top + " " + right + " " + bottom
            bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
    bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
    with open(tmp_files_path + "/gravel_predictions.json", 'w') as outfile:
        json.dump(bounding_boxes, outfile)

    count_true = 0
    predictions_file = tmp_files_path + "/gravel_predictions.json"
    predictions_data = json.load(open(predictions_file))
    nd = len(predictions_data)
    tp = [0] * nd
    fp = [0] * nd
    for idx, prediction in enumerate(predictions_data):
        file_id = prediction["file_id"]
        gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
        ground_truth_data = json.load(open(gt_file))
        ovmax = -1
        gt_match = -1
        bb = [float(x) for x in prediction["bbox"].split()]
        for obj in ground_truth_data:
            bb_gt = [float(x) for x in obj["bbox"].split()]
            bi = [max(bb[0], bb_gt[0]), max(bb[1], bb_gt[1]), min(bb[2], bb_gt[2]), min(bb[3], bb_gt[3])]
            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1
            if iw > 0 and ih > 0:
                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bb_gt[2] - bb_gt[0] + 1) * (bb_gt[3] - bb_gt[1] + 1) - iw * ih
                ov = iw * ih / ua
                if ov > ovmax:
                    ovmax = ov
                    gt_match = obj
        if ovmax >= iou_thresh:
            if not bool(gt_match["used"]):
                tp[idx] = 1
                gt_match["used"] = True
                count_true += 1
                with open(gt_file, 'w') as f:
                    f.write(json.dumps(ground_truth_data))
            else:
                fp[idx] = 1
        else:
            fp[idx] = 1
    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val
    rec = [0.0] * len(tp)
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_counter
    prec = [0.0] * len(tp)
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    ap = voc_ap(rec, prec)

    shutil.rmtree(tmp_files_path)

    pred_counter = 0
    for txt_file in predicted_files_list:
        lines_list = file_lines_to_list(txt_file)
        pred_counter += len(lines_list)

    gt_num = gt_counter
    tp = count_true
    fp = pred_counter - count_true

    p = tp / (tp + fp)
    r = tp / gt_num
    f1 = 2 * p * r / (p + r)

    return ap, gt_num, tp, fp, p, r, f1
