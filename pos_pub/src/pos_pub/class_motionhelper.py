import os, time, sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('agg')
# from pynput import keyboard
write = sys.stdout.write


class tracker(object):

    def __init__(self, _goal_list, _repeat_flag, _VERBOSE=False,
        _PURSE_THRESHOLD_SEC=2.0):
        # CONFIGURATION
        self.STEP           = 0
        self.STEP_PREV      = 0
        self.time_start     = datetime.now()
        self.GOAL_THRESHOLD = 20
        self.REPEAT_FLAG    = _repeat_flag
        self.VERBOSE        = _VERBOSE
        self.sec_pursue_threshold = _PURSE_THRESHOLD_SEC
        # SET GOAL LIST
        self.set_goal_list(_goal_list)
        print ("TRACKER READY")

    """
        SET GOAL LIST
    """
    def set_goal_list(self, _goal_list):
        self.goal_list = _goal_list
        self.ngoal     = len(self.goal_list)
        self.gidx      = 0

    """
        INITIALIZE
    """
    def start(self):
        self.STEP           = 0
        self.STEP_PREV      = 0
        self.time_start     = datetime.now()

    def is_finished(self):
        if self.STEP == 100:
            return True
        else:
            return False
    """
        TRACK
    """
    def do_track(self, _curr_pos):
        self.time_diff = datetime.now() - self.time_start
        self.sec  = np.asarray(self.time_diff.total_seconds())

        # GET CURRENT POSITION
        self.curr_pos  = np.asarray(_curr_pos)
        # INITIALIZE INTERMEDIATE GOAL AS ITS CURRENT POSITION
        self.intm_goal = self.curr_pos
        """
            FINITE STATE MACHINE STARTS HERE
        """
        if self.STEP == 0:
            """ INITIAL STEP """
            # SET CURRENT GOAL AS THE FIRST ONE IN THE LIST
            self.curr_goal = np.asarray(self.goal_list[0])
            self.STEP = 10
        elif self.STEP == 10:
            """ SET CURRENT GOAL """
            # UPDATE CURRENT GOAL BASED ON GOAD INDEX (GIDX)
            self.curr_goal = np.asarray(self.goal_list[self.gidx])
            self.STEP = 20
            self.sec_pursue_start = self.sec
            if self.VERBOSE:
                print ("CURR GOAL IS [%d][%s]"
                    % (self.gidx, self.curr_goal))
        elif self.STEP == 20:
            """ PURSUIT THE GOAL """
            self.sec_pursue_elps = self.sec - self.sec_pursue_start
            # COMPUTE GOAL DIFFERENCE
            self.goal_diff = self.curr_goal - self.curr_pos
            # CHECK GOAL REACHED,
            if np.abs(self.goal_diff).max() <= self.GOAL_THRESHOLD:
                self.STEP = 30

            # SET INTERMEDIATE GOAL
            self.USE_BANGNBANG = False
            if self.USE_BANGNBANG:
                # BANG BANG CONTROL
                MAX_DIFF = 10
                self.intm_goal = self.curr_pos + MAX_DIFF*np.sign(self.goal_diff)
            else:
                # TRUNCATED CONTROL
                GAIN     = 1
                MAX_DIFF = 50
                self.intm_goal = self.curr_pos + np.clip(GAIN*self.goal_diff,
                                    -MAX_DIFF, +MAX_DIFF)

            # IF IT PURSUED SOME POINT MORE THAN THRESHOLD,
            if self.sec_pursue_elps > self.sec_pursue_threshold:
                print ("PURSUED TOO LONG [%.2fsec]" % (self.sec_pursue_elps))
                self.STEP = 30
        elif self.STEP == 30:
            """ GOAL REACHED """
            # INCREASE GOAL INDEX AS WE'VE REACHED THE GOAL
            self.gidx = self.gidx + 1
            if self.REPEAT_FLAG:
                # IF WE WANT TO REPEAT
                if self.gidx >= self.ngoal:
                    self.gidx = 0
                self.STEP = 10
                if self.VERBOSE:
                    print ("GOAL REACHED. NEXT GIDX IS [%d]" % (self.gidx))
            else:
                # OTHERWISE,
                if self.gidx >= self.ngoal:
                    self.STEP = 100
                    if self.VERBOSE:
                        print ("GOAL REACHED. FINISH TRACKING." )
                else:
                    self.STEP = 10
                    if self.VERBOSE:
                        print ("GOAL REACHED. NEXT GIDX IS [%d]" % (self.gidx))
        elif self.STEP == 100:
            """ FINISHED """
            # STOP HERE
            DO_NOTHING = ''
        else:
            print ("UNKOWN STEP:[%d]" % (self.STEP))

        # PRINT STEP TRANSITIONS
        if self.STEP != self.STEP_PREV:
            if self.VERBOSE:
                print ("[%d]=>[%d]" % (self.STEP_PREV, self.STEP))
            self.STEP_PREV = self.STEP

        # RETURN INTERMEDIATE GOAL
        return self.intm_goal, self.curr_goal


class timer(object):
    def __init__(self,_HZ, _MAX_SEC,_VERBOSE=True):
        self.time_start     = datetime.now()
        self.sec_next       = 0.0
        self.HZ             = _HZ
        if self.HZ == 0:
            self.sec_period  = 0
        else:
            self.sec_period  = 1.0 / self.HZ
        self.max_sec        = _MAX_SEC
        self.sec_elps       = 0.0
        self.sec_elps_prev  = 0.0
        self.sec_elps_diff  = 0.0
        self.tick           = 0.
        self.force_finish   = False
        self.DELAYED_FLAG   = False
        self.VERBOSE        = _VERBOSE
        print ("TIMER WITH [%d]HZ INITIALIZED. MAX_SEC IS [%.1fsec]."
            % (self.HZ, self.max_sec))

    def start(self):
        self.time_start     = datetime.now()
        self.sec_next       = 0.0
        self.sec_elps       = 0.0
        self.sec_elps_prev  = 0.0
        self.sec_elps_diff  = 0.0
        self.tick           = 0.

    def finish(self):
        self.force_finish = True

    def is_finished(self):
        self.time_diff = datetime.now() - self.time_start
        self.sec_elps  = self.time_diff.total_seconds()
        if self.force_finish:
            return True
        if self.sec_elps > self.max_sec:
            return True
        else:
            return False

    def is_notfinished(self):
        self.time_diff = datetime.now() - self.time_start
        self.sec_elps  = self.time_diff.total_seconds()
        if self.force_finish:
            return False
        if self.sec_elps > self.max_sec:
            return False
        else:
            return True

    def do_run(self):
        self.time_diff = datetime.now() - self.time_start
        self.sec_elps  = self.time_diff.total_seconds()
        if self.sec_elps > self.sec_next:
            self.sec_next = self.sec_next + self.sec_period
            self.tick     = self.tick + 1
            """ COMPUTE THE TIME DIFFERENCE & UPDATE PREVIOUS ELAPSED TIME """
            self.sec_elps_diff = self.sec_elps - self.sec_elps_prev
            self.sec_elps_prev = self.sec_elps
            """ CHECK DELAYED """
            if (self.sec_elps_diff > self.sec_period*1.5) & (self.HZ != 0):
                if self.VERBOSE:
                    print ("sec_elps_diff:[%.1fms]" % (self.sec_elps_diff*1000.0))
                    print ("[%d][%.1fs] DELAYED! T:[%.1fms] BUT IT TOOK [%.1fms]"
                        % (self.tick, self.sec_elps, self.sec_period*1000.0, self.sec_elps_diff*1000.0))
                self.DELAYED_FLAG = True
            return True
        else:
            return False

class kbd_handler(object):

    def __init__(self):
        # INITIALIZE KEYBOARD
        self.key = ''
        lis = keyboard.Listener(on_press=self.on_press)
        lis.start()
        print ("INSTANTIATING KEYBOARD HANDLER")

    def on_press(self, key):
        # ON PRESS FUNCTION
        try:    k = key.char # single-char keys
        except: k = key.name # other keys
        self.key = k

        # REMOVE CURRENT CHARACTER
        write('\b \b')
        # print ("[%s] PRESSED." % (self.key))

    def is_pressed(self, _key):
        if self.key == _key:
            return True
        else:
            return False

    def reset(self):
        self.key = ''

def plot_arrow(_x, _y, _d, _len=10.,_col='r'):
    if _len==0: return
    r = _d*np.pi/180.
    w = _len
    xlen,ylen = w*np.cos(r), w*np.sin(r)
    plt.arrow(_x, _y, xlen, ylen,
              fc=_col, ec='k', alpha=0.5, width=_len/2, head_width=_len,
              head_length=_len)

def plot_traj(posconcat, _figsize=(9,9)):
    plt.figure(figsize=_figsize)
    plt.plot(posconcat[:,0],posconcat[:,1],'g',lw=3)
    for i in range(posconcat.shape[0]-1):
        p = posconcat[i, :]
        pnext = posconcat[i+1, :]
        dist = np.linalg.norm(p[0:2]-pnext[0:2])/2
        if dist==0: continue
        plot_arrow(p[0],p[1],p[2],_len=dist,_col='b')
        plt.text(p[0],p[1],'%d'%(i),fontsize=15)
    p = posconcat[0, :]
    plot_arrow(p[0],p[1],p[2],_len=30,_col='r')
    p = posconcat[-1, :]
    plot_arrow(p[0],p[1],p[2],_len=30,_col='m')
    plt.axis('equal')

























#
