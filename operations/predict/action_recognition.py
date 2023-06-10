from abc import abstractmethod
import cv2
import math
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import torch
from tqdm import tqdm

from .base import BasePredictor
from .ar_tools import inference_detector, inference_top_down_pose_model, det_test_pipeline

from mmdet.apis import init_detector
from mmpose.apis import init_pose_model

det_config = 'tools/mmdet_configs/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'
det_checkpoint = '../pretrained/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
pose_config = 'tools/mmpose_configs/hrnet_w32_coco_256x192.py'
pose_checkpoint = '../pretrained/hrnet_w32_coco_256x192-c78dce93_20200708.pth'

# Font file
# font_path = 'C:/Windows/Fonts/msyh.ttc'
font_path = '../temp/msyh.ttc'

hmdb51_classes = [
    'brush_hair', 'cartwheel', 'catch', 'chew', 'clap',
    'climb', 'climb_stairs', 'dive', 'draw_sword', 'dribble',
    'drink', 'eat', 'fall_floor', 'fencing', 'flic_flac',
    'golf', 'handstand', 'hit', 'hug', 'jump',
    'kick', 'kick_ball', 'kiss', 'laugh', 'pick',
    'pour', 'pullup', 'punch', 'push', 'pushup',
    'ride_bike', 'ride_horse', 'run', 'shake_hands', 'shoot_ball',
    'shoot_bow', 'shoot_gun', 'sit', 'situp', 'smile',
    'smoke', 'somersault', 'stand', 'swing_baseball', 'sword',
    'sword_exercise', 'talk', 'throw', 'turn', 'walk',
    'wave'
]
ucf101_classes = [
    'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
    'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress',
    'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats',
    'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth',
    'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen',
    'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics',
    'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'HammerThrow',
    'Hammering', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump',
    'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow',
    'JugglingBalls', 'JumpRope', 'JumpingJack', 'Kayaking', 'Knitting',
    'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor',
    'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf',
    'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar',
    'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps',
    'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing',
    'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding',
    'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty',
    'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot',
    'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing',
    'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard',
    'YoYo',
]
kinetics400_classes = [
    "abseiling", "air_drumming", "answering_questions", "applauding", "applying_cream",
    "archery", "arm_wrestling", "arranging_flowers", "assembling_computer", "auctioning",
    "baby_waking_up", "baking_cookies", "balloon_blowing", "bandaging", "barbequing",
    "bartending", "beatboxing", "bee_keeping", "belly_dancing", "bench_pressing",
    "bending_back", "bending_metal", "biking_through_snow", "blasting_sand", "blowing_glass",
    "blowing_leaves", "blowing_nose", "blowing_out_candles", "bobsledding", "bookbinding",
    "bouncing_on_trampoline", "bowling", "braiding_hair", "breading_or_breadcrumbing", "breakdancing",
    "brush_painting", "brushing_hair", "brushing_teeth", "building_cabinet", "building_shed",
    "bungee_jumping", "busking", "canoeing_or_kayaking", "capoeira", "carrying_baby",
    "cartwheeling", "carving_pumpkin", "catching_fish", "catching_or_throwing_baseball", "catching_or_throwing_frisbee",
    "catching_or_throwing_softball", "celebrating", "changing_oil", "changing_wheel", "checking_tires",
    "cheerleading", "chopping_wood", "clapping", "clay_pottery_making", "clean_and_jerk",
    "cleaning_floor", "cleaning_gutters", "cleaning_pool", "cleaning_shoes", "cleaning_toilet",
    "cleaning_windows", "climbing_a_rope", "climbing_ladder", "climbing_tree", "contact_juggling",
    "cooking_chicken", "cooking_egg", "cooking_on_campfire", "cooking_sausages", "counting_money",
    "country_line_dancing", "cracking_neck", "crawling_baby", "crossing_river", "crying",
    "curling_hair", "cutting_nails", "cutting_pineapple", "cutting_watermelon", "dancing_ballet",
    "dancing_charleston", "dancing_gangnam_style", "dancing_macarena", "deadlifting", "decorating_the_christmas_tree",
    "digging", "dining", "disc_golfing", "diving_cliff", "dodgeball",
    "doing_aerobics", "doing_laundry", "doing_nails", "drawing", "dribbling_basketball",
    "drinking", "drinking_beer", "drinking_shots", "driving_car", "driving_tractor",
    "drop_kicking", "drumming_fingers", "dunking_basketball", "dying_hair", "eating_burger",
    "eating_cake", "eating_carrots", "eating_chips", "eating_doughnuts", "eating_hotdog",
    "eating_ice_cream", "eating_spaghetti", "eating_watermelon", "egg_hunting", "exercising_arm",
    "exercising_with_an_exercise_ball", "extinguishing_fire", "faceplanting", "feeding_birds", "feeding_fish",
    "feeding_goats", "filling_eyebrows", "finger_snapping", "fixing_hair", "flipping_pancake",
    "flying_kite", "folding_clothes", "folding_napkins", "folding_paper", "front_raises",
    "frying_vegetables", "garbage_collecting", "gargling", "getting_a_haircut", "getting_a_tattoo",
    "giving_or_receiving_award", "golf_chipping", "golf_driving", "golf_putting", "grinding_meat",
    "grooming_dog", "grooming_horse", "gymnastics_tumbling", "hammer_throw", "headbanging",
    "headbutting", "high_jump", "high_kick", "hitting_baseball", "hockey_stop",
    "holding_snake", "hopscotch", "hoverboarding", "hugging", "hula_hooping",
    "hurdling", "hurling_(sport)", "ice_climbing", "ice_fishing", "ice_skating",
    "ironing", "javelin_throw", "jetskiing", "jogging", "juggling_balls",
    "juggling_fire", "juggling_soccer_ball", "jumping_into_pool", "jumpstyle_dancing", "kicking_field_goal",
    "kicking_soccer_ball", "kissing", "kitesurfing", "knitting", "krumping",
    "laughing", "laying_bricks", "long_jump", "lunge", "making_a_cake",
    "making_a_sandwich", "making_bed", "making_jewelry", "making_pizza", "making_snowman",
    "making_sushi", "making_tea", "marching", "massaging_back", "massaging_feet",
    "massaging_legs", "massaging_person's_head", "milking_cow", "mopping_floor", "motorcycling",
    "moving_furniture", "mowing_lawn", "news_anchoring", "opening_bottle", "opening_present",
    "paragliding", "parasailing", "parkour", "passing_American_football_(in_game)",
    "passing_American_football_(not_in_game)",
    "peeling_apples", "peeling_potatoes", "petting_animal_(not_cat)", "petting_cat", "picking_fruit",
    "planting_trees", "plastering", "playing_accordion", "playing_badminton", "playing_bagpipes",
    "playing_basketball", "playing_bass_guitar", "playing_cards", "playing_cello", "playing_chess",
    "playing_clarinet", "playing_controller", "playing_cricket", "playing_cymbals", "playing_didgeridoo",
    "playing_drums", "playing_flute", "playing_guitar", "playing_harmonica", "playing_harp",
    "playing_ice_hockey", "playing_keyboard", "playing_kickball", "playing_monopoly", "playing_organ",
    "playing_paintball", "playing_piano", "playing_poker", "playing_recorder", "playing_saxophone",
    "playing_squash_or_racquetball", "playing_tennis", "playing_trombone", "playing_trumpet", "playing_ukulele",
    "playing_violin", "playing_volleyball", "playing_xylophone", "pole_vault", "presenting_weather_forecast",
    "pull_ups", "pumping_fist", "pumping_gas", "punching_bag", "punching_person_(boxing)",
    "push_up", "pushing_car", "pushing_cart", "pushing_wheelchair", "reading_book",
    "reading_newspaper", "recording_music", "riding_a_bike", "riding_camel", "riding_elephant",
    "riding_mechanical_bull", "riding_mountain_bike", "riding_mule", "riding_or_walking_with_horse", "riding_scooter",
    "riding_unicycle", "ripping_paper", "robot_dancing", "rock_climbing", "rock_scissors_paper",
    "roller_skating", "running_on_treadmill", "sailing", "salsa_dancing", "sanding_floor",
    "scrambling_eggs", "scuba_diving", "setting_table", "shaking_hands", "shaking_head",
    "sharpening_knives", "sharpening_pencil", "shaving_head", "shaving_legs", "shearing_sheep",
    "shining_shoes", "shooting_basketball", "shooting_goal_(soccer)", "shot_put", "shoveling_snow",
    "shredding_paper", "shuffling_cards", "side_kick", "sign_language_interpreting", "singing",
    "situp", "skateboarding", "ski_jumping", "skiing_(not_slalom_or_crosscountry)", "skiing_crosscountry",
    "skiing_slalom", "skipping_rope", "skydiving", "slacklining", "slapping",
    "sled_dog_racing", "smoking", "smoking_hookah", "snatch_weight_lifting", "sneezing",
    "sniffing", "snorkeling", "snowboarding", "snowkiting", "snowmobiling",
    "somersaulting", "spinning_poi", "spray_painting", "spraying", "springboard_diving",
    "squat", "sticking_tongue_out", "stomping_grapes", "stretching_arm", "stretching_leg",
    "strumming_guitar", "surfing_crowd", "surfing_water", "sweeping_floor", "swimming_backstroke",
    "swimming_breast_stroke", "swimming_butterfly_stroke", "swing_dancing", "swinging_legs", "swinging_on_something",
    "sword_fighting", "tai_chi", "taking_a_shower", "tango_dancing", "tap_dancing",
    "tapping_guitar", "tapping_pen", "tasting_beer", "tasting_food", "testifying",
    "texting", "throwing_axe", "throwing_ball", "throwing_discus", "tickling",
    "tobogganing", "tossing_coin", "tossing_salad", "training_dog", "trapezing",
    "trimming_or_shaving_beard", "trimming_trees", "triple_jump", "tying_bow_tie", "tying_knot_(not_on_a_tie)",
    "tying_tie", "unboxing", "unloading_truck", "using_computer", "using_remote_controller_(not_gaming)",
    "using_segway", "vault", "waiting_in_line", "walking_the_dog", "washing_dishes",
    "washing_feet", "washing_hair", "washing_hands", "water_skiing", "water_sliding",
    "watering_plants", "waxing_back", "waxing_chest", "waxing_eyebrows", "waxing_legs",
    "weaving_basket", "welding", "whistling", "windsurfing", "wrapping_present",
    "wrestling", "writing", "yawning", "yoga", "zumba",
]

l_pair = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (17, 11), (17, 12),  # Body
    (11, 13), (12, 14), (13, 15), (14, 16)
]

p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
           # Nose, LEye, REye, LEar, REar
           (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
           # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
           (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
           (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck

line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
              (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
              (77, 222, 255), (255, 156, 127),
              (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]


class ActionRecognitionPredictor(BasePredictor):
    """
    Base class of all action recognition predictors.
    """

    def __init__(self,
                 inputs: list,
                 transform,
                 model: torch.nn.Module,
                 device=None,
                 show_output=True,
                 output_dir=None,
                 clip_len=128,
                 clip_interval=30,
                 clip_len_s=None,
                 clip_interval_s=None,
                 top_k=5):
        super(ActionRecognitionPredictor, self).__init__(
            inputs, transform, model, device, show_output, output_dir
        )

        assert clip_len > 0 and isinstance(clip_len, int)
        assert clip_interval > 0 and isinstance(clip_interval, int)
        assert clip_len_s is None or clip_len_s > 0
        assert clip_interval_s is None or clip_interval_s > 0
        assert top_k > 0 and isinstance(top_k, int)
        self.clip_len = clip_len
        self.clip_interval = clip_interval
        self.clip_len_s = clip_len_s
        self.clip_interval_s = clip_interval_s
        self.top_k = top_k

        # pil settings
        self.title_color = (0, 153, 255)  # blue, RGB
        self.top1_color = (255, 0, 0)  # red, RGB
        self.top2_color = (255, 165, 0)  # orange, RGB
        self.top_k_color = (0, 224, 0)
        self.bbox_color = (0, 255, 0)
        self.p_color = p_color
        self.line_color = line_color
        self.min_font_size = 10
        self.font_size_to_video_wh = 0.04
        self.font_size_to_video_w_max = 0.06
        self.font = font_path
        self.class_max_len = 25
        self.classes_ul_pos = (0.05, 0.05)  # w, h
        self.line_space = 0.2

    def _read_video(self, path, return_frames=False):
        """ uses cv2.VideoCapture (reads BGR video, and convert it to RGB) """
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert fps > 0, f'Input video "{path}" has a non-positive fps'
        if self.clip_len_s is not None:
            self.clip_len = max(round(fps * self.clip_len_s), 1)
        if self.clip_interval_s is not None:
            self.clip_interval = max(round(fps * self.clip_interval_s), 1)

        frames = []
        ret, frame = cap.read()
        while ret:
            frames.append(frame)
            ret, frame = cap.read()
        cap.release()

        if return_frames:
            frames = [np.flip(frame, axis=2) for frame in frames]
            return frames, fps

        if len(frames) > 0:
            video = torch.from_numpy(np.stack(frames))
        else:
            video = torch.empty((0, 1, 1, 3), dtype=torch.uint8)
        video = torch.flip(video, dims=(3,))
        return video, fps

    # def _read_video(self, path, return_frames=False):
    #     """ uses pyav (reads RGB video) """
    #     from .tools import read_video
    #     video, fps = read_video(path, return_frames=return_frames)
    #     fps = float(fps)
    #     assert fps > 0, f'Input video "{path}" has a non-positive fps'
    #     if self.clip_len_s is not None:
    #         self.clip_len = max(round(fps * self.clip_len_s), 1)
    #     if self.clip_interval_s is not None:
    #         self.clip_interval = max(round(fps * self.clip_interval_s), 1)
    #
    #     return video, fps

    def _clip_video(self, video: torch.Tensor):
        clips = []
        frame_ranges = []
        last_s = len(video) - self.clip_len
        if last_s <= 0:
            clips.append(video)
            frame_ranges.append((0, len(video)))
        else:
            num_clips = math.ceil(last_s / self.clip_interval)
            for i in range(num_clips):
                s = self.clip_interval * i
                e = s + self.clip_len
                clips.append(video[s:e])
                frame_ranges.append((s, e))

        return clips, frame_ranges

    def _pil_draw_results(self,
                          frame: np.ndarray,
                          results: dict,
                          **kwargs):
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        image_size_hw = frame.shape[:2]

        classes = results.get('classes')
        probs = results.get('probs')
        bbox = results.get('bbox')
        kpt = results.get('kpt')
        kpt_score = results.get('kpt_score')

        draw_classes = kwargs.get('draw_classes')
        draw_bbox = kwargs.get('draw_bbox')
        draw_skeleton = kwargs.get('draw_skeleton')

        if draw_bbox and bbox is not None:
            self._pil_draw_bbox(draw, bbox)
        if draw_skeleton and kpt is not None:
            self._pil_draw_skeleton(draw, kpt, kpt_score)
        if draw_classes and (classes is not None and probs is not None):
            self._pil_draw_classes_and_probs(draw, classes, probs, image_size_hw)

        image = np.array(pil_image)
        return image

    def _pil_draw_bbox(self, draw, bbox):
        """
        bbox: (x1, y1, x2, y2, ...)
        """
        for b in bbox:
            x1, y1, x2, y2 = [int(x) for x in b[:4]]
            bbox_pos = (x1, y1, x2, y2)
            w = max(x2 - x1, 1)
            rect_w = max(round(w * 0.02), 1)

            draw.rectangle(bbox_pos, None, self.bbox_color, rect_w)

    def _pil_draw_skeleton(self, draw, kpt, kpt_score=None):
        if len(kpt) == 0:
            return

        kp_num = len(kpt[0])
        assert kp_num == 17

        if kpt_score is None:
            kpt_score = np.ones((len(kpt), kp_num))

        for i in range(min(len(kpt), len(kpt_score))):
            part_line = {}

            pred = kpt[i]
            score = kpt_score[i]

            # the "18th point" (midpoint of the two shoulders)
            pred_mid = (pred[5, :] + pred[6, :]) / 2
            score_mid = (score[5] + score[6]) / 2
            pred = np.concatenate((pred, pred_mid[np.newaxis, ...]))
            score = np.concatenate((score, score_mid[np.newaxis, ...]))

            width = pred[:, 0].max() - pred[:, 0].min()
            rad = max(round(width * 0.02), 1)
            th = max(round(width * 0.015), 1)

            vis_thres = 0.4

            # Draw keypoints
            num_draw_kp = kp_num
            for j in range(num_draw_kp):
                if score[j] <= vis_thres:
                    continue
                cor_x, cor_y = int(pred[j, 0]), int(pred[j, 1])
                part_line[j] = (cor_x, cor_y)
                color = self.p_color[j] if j < len(p_color) else (255, 255, 255)
                draw.ellipse((cor_x - rad, cor_y - rad, cor_x + rad, cor_y + rad), color, rad)

            # Draw limbs
            for j, (start_p, end_p) in enumerate(l_pair):
                if start_p in part_line and end_p in part_line:
                    start_xy = part_line[start_p]
                    end_xy = part_line[end_p]
                    color = self.line_color[j] if j < len(line_color) else (255, 255, 255)
                    draw.line(start_xy + end_xy, color, th)

    def _pil_draw_classes_and_probs(self, draw, classes, probs, image_size_hw):
        h, w = image_size_hw
        ul_pos = round(w * self.classes_ul_pos[0]), round(h * self.classes_ul_pos[1])

        font_size = max(round(max(w, h) * self.font_size_to_video_wh), self.min_font_size)
        font_size = min(round(w * self.font_size_to_video_w_max), font_size)
        font = ImageFont.truetype(self.font, font_size)
        lines_colors = [('Possible actions:', self.title_color)]
        line_height = font.getsize('A')[1]
        ul_pos = (ul_pos[0], ul_pos[1] + round(line_height * self.line_space / 2))
        uls_heights = [(ul_pos, line_height)]
        ul_pos = (ul_pos[0], ul_pos[1] + line_height + round(line_height * self.line_space / 2))
        for i, (cls, prob) in enumerate(zip(classes, probs)):
            if len(cls) > self.class_max_len:
                cls = cls[:self.class_max_len - 3] + '...'
            prob = "%.1f" % (prob * 100)
            line = f'{cls}: {prob}%'
            if i == 0:
                color = self.top1_color
            elif i == 1:
                color = self.top2_color
            else:
                color = self.top_k_color
            lines_colors.append((line, color))
            ul_pos = (ul_pos[0], ul_pos[1] + round(line_height * self.line_space / 2))
            uls_heights.append((ul_pos, line_height))
            ul_pos = (ul_pos[0], ul_pos[1] + line_height + round(line_height * self.line_space / 2))
        max_line_idx = 0
        for i in range(len(uls_heights)):
            ul_height = uls_heights[i]
            if ul_height[0][1] + ul_height[1] > h:
                break
            max_line_idx += 1
        for i in range(max_line_idx):
            line, color = lines_colors[i]
            ul = uls_heights[i][0]
            draw.text(ul, line, color, font)

    @staticmethod
    def _save_video(path, video, fps):
        """ uses cv2.VideoWriter (saves a video in BGR) """
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = video.shape[2], video.shape[1]
        out = cv2.VideoWriter(path, fourcc, fps, size)
        for frame in video:
            out.write(frame)
        out.release()

    @abstractmethod
    def predict(self):
        pass


class UCF101Predictor(ActionRecognitionPredictor):
    """
    UCF101 predictor.
    Args:
        inputs: Paths of input video files.
        transform: Transformations.
        model: Model.
        device: Device.
        show_output: Whether to output video with marked results.
        output_dir: Output directory.
        clip_len: Length (in number of frames) of a single clip from the input
            file. Default: 128.
        clip_interval: Interval (in number of frames) between two clips.
            Default: 30.
        clip_len_s: Length (in second) of a single clip from the input file. If
            specified, cover clip_len. Default: None.
        clip_interval_s: Interval (in second) between two clips. If
            specified, cover clip_interval. Default: None.
        batch_size: Batch size for the model. Default: 2.
        top_k: "k" of the top-k most possible classes to show. Default: 5.
    """

    classes = ucf101_classes
    save_file_suffix = '-ucf101-prediction'

    def __init__(self,
                 inputs: list,
                 transform,
                 model: torch.nn.Module,
                 device=None,
                 show_output=True,
                 output_dir=None,
                 clip_len=128,
                 clip_interval=30,
                 clip_len_s=None,
                 clip_interval_s=None,
                 batch_size=2,
                 top_k=5):
        super(UCF101Predictor, self).__init__(
            inputs, transform, model, device, show_output, output_dir,
            clip_len, clip_interval, clip_len_s, clip_interval_s, top_k
        )
        assert batch_size > 0 and isinstance(batch_size, int)
        self.batch_size = batch_size

        # cv2 settings
        self.show_max_w = 1280
        self.show_max_h = 720

    def _draw_results(self, video: np.ndarray, logits, frame_ranges) -> np.ndarray:
        """
        len(logits) = len(frame_ranges)
        """
        video = video.copy()

        # get top-k
        top_k_list = []
        prob_list = []
        for logit in logits:
            softmax = np.exp(logit)
            softmax = softmax / softmax.sum()
            idx_rank = np.flip(np.argsort(logit))
            top_k = idx_rank[:self.top_k].tolist()
            prob = [softmax[i] for i in top_k]
            top_k_list.append(top_k)
            prob_list.append(prob)

        # get show table
        show_table = -np.ones(len(video), dtype=np.int32)
        for i, r in enumerate(frame_ranges):
            s = round((r[0] + r[1]) / 2)
            e = r[1] + self.clip_interval
            show_table[s:e] = i

        # draw results
        for i in range(len(video)):
            show_idx = show_table[i]
            if show_idx == -1:
                continue
            top_k = top_k_list[show_idx]
            classes = [self.classes[x] for x in top_k]
            probs = prob_list[show_idx]
            results = {
                'classes': classes,
                'probs': probs,
            }
            video[i] = self._pil_draw_results(video[i], results, draw_classes=True)

        return video

    def predict(self):
        print('Predicting...')
        self.model.eval()

        save = self.output_dir is not None
        bar = tqdm(total=len(self.inputs)) if not self.show_output else None
        for i, file_path in enumerate(self.inputs):
            if self.show_output:
                print(f'Predicting {i + 1}/{len(self.inputs)}: {file_path}')
            video, fps = self._read_video(file_path)
            if len(video) == 0:
                print(f'Video in {file_path} is empty. Skip.')
                bar.update()
                continue

            clips, frame_ranges = self._clip_video(video)

            logits = []
            num_batches = math.ceil(len(clips) / self.batch_size)
            for bi in range(num_batches):
                clips_batch = clips[bi * self.batch_size:(bi + 1) * self.batch_size]
                inp = []
                for clip in clips_batch:
                    inp.append(self.transform(clip))
                inp = torch.stack(inp)
                if len(inp.shape) == 6:  # (N,num_seg,C,T,H,W)
                    tmp_logits = []
                    for j in range(inp.shape[1]):
                        inp_seg = inp[:, j].to(self.device)
                        out = self.model(inp_seg)
                        logit = out.detach().cpu().numpy()
                        tmp_logits.append(logit)
                    logit = np.stack(tmp_logits).mean(axis=0)[0]
                else:  # (N,C,T,H,W)
                    inp = inp.to(self.device)
                    out = self.model(inp)
                    logit = out.detach().cpu().numpy()[0]
                logits.append(logit)

            result = self._draw_results(video.numpy(), logits, frame_ranges)
            result = np.flip(result, axis=-1)  # cv2.imshow should accept a BGR image, not RGB!

            # show result
            if self.show_output:
                print('Showing result...')
                show_size = result.shape[2], result.shape[1]
                resize_flag = False
                if show_size[0] > self.show_max_w:
                    show_size = (self.show_max_w, round(show_size[1] * self.show_max_w / show_size[0]))
                    resize_flag = True
                if show_size[1] > self.show_max_h:
                    show_size = (round(show_size[0] * self.show_max_h / show_size[1]), self.show_max_h)
                    resize_flag = True
                wait_ms = round(1000 / fps)

                for j in range(len(result)):
                    frame = result[j]
                    if resize_flag:
                        frame = cv2.resize(frame, show_size)
                    cv2.imshow(f'result of {file_path}', frame)
                    key = cv2.waitKey(wait_ms)
                    if key & 0xff == ord('q'):  # press 'q' to quit manually
                        break
                cv2.destroyAllWindows()

            # save result
            if save:
                file_name = os.path.splitext(os.path.split(file_path)[1])[0]
                output_path = os.path.join(self.output_dir, f'{file_name}{self.save_file_suffix}.avi')
                self._save_video(output_path, result, fps)
                print(f'Result saved in {output_path}.')

            if not self.show_output:
                bar.update()
        if not self.show_output:
            bar.close()


class UCF101RGBSkeletonPredictor(ActionRecognitionPredictor):
    classes = ucf101_classes
    save_file_suffix = '-ucf101-rgb_skeleton-prediction'

    def __init__(self,
                 inputs: list,
                 transform,
                 model: torch.nn.Module,
                 device=None,
                 show_output=True,
                 output_dir=None,
                 clip_len=128,
                 clip_interval=30,
                 clip_len_s=None,
                 clip_interval_s=None,
                 batch_size=2,
                 top_k=5,
                 det_batch_size=1,
                 det_score_thr=0.5,
                 draw_bbox=False,
                 draw_skeleton=False):
        super(UCF101RGBSkeletonPredictor, self).__init__(
            inputs, transform, model, device, show_output, output_dir,
            clip_len, clip_interval, clip_len_s, clip_interval_s, top_k)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert det_batch_size > 0 and isinstance(det_batch_size, int)
        assert 0 <= det_score_thr <= 1
        self.batch_size = batch_size
        self.det_batch_size = det_batch_size
        self.det_score_thr = det_score_thr
        self.draw_bbox = draw_bbox
        self.draw_skeleton = draw_skeleton

        # cv2 settings
        self.show_max_w = 1280
        self.show_max_h = 720

        print('Loading detection and pose models...')
        self.det_model = self._load_detection_model()
        self.det_test_pipeline = det_test_pipeline(self.det_model)
        self.pose_model = self._load_pose_model()
        print('Loading done.')

    def _load_detection_model(self):
        det_model = init_detector(det_config, det_checkpoint, self.device)
        assert det_model.CLASSES[0] == 'person', 'Require to use a detection model trained on COCO'
        return det_model

    def _load_pose_model(self):
        pose_model = init_pose_model(pose_config, pose_checkpoint, self.device)
        return pose_model

    def _generate_bbox_skeleton(self, frames: list):
        # generate bbox
        det_results = []
        num_batches = math.ceil(len(frames) / self.det_batch_size)

        for bi in range(num_batches):
            b_frames = frames[bi * self.det_batch_size:(bi + 1) * self.det_batch_size]
            result = inference_detector(self.det_model, b_frames, self.device, self.det_test_pipeline)
            result = [r[0][r[0][:, 4] >= self.det_score_thr] for r in result]
            det_results.extend(result)

        # generate skeleton
        num_frames = len(det_results)
        num_person = max([len(x) for x in det_results]) if num_frames > 0 else 0
        kp = np.zeros((num_person, num_frames, 17, 3), dtype=np.float32)

        for i, (f, d) in enumerate(zip(frames, det_results)):
            d = [dict(bbox=x) for x in list(d)]
            pose = inference_top_down_pose_model(self.pose_model, f, d, format='xyxy')[0]
            for j, item in enumerate(pose):
                kp[j, i] = item['keypoints']

        frame_shape = frames[0].shape[:2] if len(frames) > 0 else (0, 0)
        bbox_skeleton = {
            'bbox': det_results,
            'keypoint': kp[..., :2],
            'keypoint_score': kp[..., 2],
            'img_shape': frame_shape,
        }
        return bbox_skeleton

    def _draw_results(self,
                      video: np.ndarray,
                      logits,
                      frame_ranges,
                      bboxes,
                      kpts,
                      kpt_scores):
        """
        len(logits) = len(frame_ranges) = len(bboxes) == len(kpts) == len(kpt_scores)
        bboxes: list with len=num_frames, each: (num_persons, 5)
        kpts: (num_frames, num_persons, num_kpts, 2)
        kpt_scores: (num_frames, num_persons, num_kpts)
        """
        video = video.copy()

        # get top-k
        top_k_list = []
        prob_list = []
        for logit in logits:
            softmax = np.exp(logit - np.mean(logit))
            softmax = softmax / softmax.sum()
            idx_rank = np.flip(np.argsort(logit))
            top_k = idx_rank[:self.top_k].tolist()
            prob = [softmax[i] for i in top_k]
            top_k_list.append(top_k)
            prob_list.append(prob)

        # get show table
        show_table = -np.ones(len(video), dtype=np.int32)
        for i, r in enumerate(frame_ranges):
            s = round((r[0] + r[1]) / 2)
            e = r[1] + self.clip_interval
            show_table[s:e] = i

        # draw results
        for i in range(len(video)):
            show_idx = show_table[i]
            if show_idx == -1:
                classes, probs = None, None
            else:
                top_k = top_k_list[show_idx]
                classes = [self.classes[x] for x in top_k]
                probs = prob_list[show_idx]
            results = {
                'classes': classes,
                'probs': probs,
                'bbox': bboxes[i],
                'kpt': kpts[i],
                'kpt_score': kpt_scores[i],
            }
            video[i] = self._pil_draw_results(
                video[i], results, draw_classes=True,
                draw_bbox=self.draw_bbox, draw_skeleton=self.draw_skeleton
            )

        return video

    def predict(self):
        print('Predicting...')
        self.model.eval()

        save = self.output_dir is not None
        bar = tqdm(total=len(self.inputs)) if not self.show_output else None
        for i, file_path in enumerate(self.inputs):
            if self.show_output:
                print(f'Predicting {i + 1}/{len(self.inputs)}: {file_path}')
            frames, fps = self._read_video(file_path, return_frames=True)
            if len(frames) == 0:
                print(f'Video in {file_path} is empty. Skip.')
                bar.update()
                continue

            bbox_skeleton = self._generate_bbox_skeleton(frames)

            video = torch.from_numpy(np.stack(frames))
            clips, frame_ranges = self._clip_video(video)

            logits = []
            num_batches = math.ceil(len(clips) / self.batch_size)
            for bi in range(num_batches):
                clips_batch = clips[bi * self.batch_size:(bi + 1) * self.batch_size]
                frame_ranges_batch = frame_ranges[bi * self.batch_size:(bi + 1) * self.batch_size]
                rgb_list, skeleton_image_list = [], []
                for clip, frame_range in zip(clips_batch, frame_ranges_batch):
                    s, e = frame_range
                    data = {
                        'rgb': clip,
                        'keypoint': bbox_skeleton['keypoint'][:, s:e],
                        'keypoint_score': bbox_skeleton['keypoint_score'][:, s:e],
                        'total_frames': e - s,
                        'original_shape': bbox_skeleton['img_shape'],
                        'img_shape': bbox_skeleton['img_shape'],
                        'start_index': 0,
                        'label': -1,
                    }
                    rgb, skeleton_image, label = self.transform(data)
                    rgb_list.append(rgb)
                    skeleton_image_list.append(skeleton_image)
                inp = (torch.stack(rgb_list), torch.stack(skeleton_image_list))
                if len(inp[0].shape) == 6:  # (N,num_seg,C,T,H,W)
                    tmp_logits = []
                    for j in range(inp[0].shape[1]):
                        inp_seg = [item[:, j].to(self.device) for item in inp if isinstance(item, torch.Tensor)]
                        out = self.model(inp_seg)
                        logit = out.detach().cpu().numpy()
                        tmp_logits.append(logit)
                    logit = np.stack(tmp_logits).mean(axis=0)[0]
                else:  # (N,C,T,H,W)
                    inp = [item.to(self.device) for item in inp if isinstance(item, torch.Tensor)]
                    out = self.model(inp)
                    logit = out.detach().cpu().numpy()[0]
                logits.append(logit)

            result = self._draw_results(video.numpy(), logits, frame_ranges,
                                        bboxes=bbox_skeleton['bbox'],
                                        kpts=bbox_skeleton['keypoint'].transpose(1, 0, 2, 3),
                                        kpt_scores=bbox_skeleton['keypoint_score'].transpose(1, 0, 2))
            result = np.flip(result, axis=-1)  # cv2.imshow should accept a BGR image, not RGB!

            # show result
            if self.show_output:
                print('Showing result...')
                show_size = result.shape[2], result.shape[1]
                resize_flag = False
                if show_size[0] > self.show_max_w:
                    show_size = (self.show_max_w, round(show_size[1] * self.show_max_w / show_size[0]))
                    resize_flag = True
                if show_size[1] > self.show_max_h:
                    show_size = (round(show_size[0] * self.show_max_h / show_size[1]), self.show_max_h)
                    resize_flag = True
                wait_ms = round(1000 / fps)

                for j in range(len(result)):
                    frame = result[j]
                    if resize_flag:
                        frame = cv2.resize(frame, show_size)
                    cv2.imshow(f'result of {file_path}', frame)
                    key = cv2.waitKey(wait_ms)
                    if key & 0xff == ord('q'):  # press 'q' to quit manually
                        break
                cv2.destroyAllWindows()

            # save result
            if save:
                file_name = os.path.splitext(os.path.split(file_path)[1])[0]
                output_path = os.path.join(self.output_dir, f'{file_name}{self.save_file_suffix}.avi')
                self._save_video(output_path, result, fps)
                print(f'Result saved in {output_path}.')

            if not self.show_output:
                bar.update()
        if not self.show_output:
            bar.close()


class UCF101TinyRGBSkeletonPredictor(UCF101RGBSkeletonPredictor):
    classes = [
        'ApplyEyeMakeup', 'ApplyLipStick', 'Basketball', 'BasketballDunk', 'BenchPress',
        'Biking', 'Billiards', 'BlowDryHair', 'BodyWeightSquats', 'BrushingTeeth',
        'HeadMessage', 'JumpingJack', 'JumpRope', 'MoppingFloor', 'Pullups',
        'Pushups', 'ShavingBeard', 'Typing', 'WallPushups', 'WritingOnBoard',
    ]
    save_file_suffix = '-ucf101-tiny-rgb_skeleton-prediction'
