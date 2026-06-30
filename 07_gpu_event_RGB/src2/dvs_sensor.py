# Damien JOUBERT 17-01-2020 - Updated by AvS 23-02-2024
import numpy as np
from .event_buffer import EventBuffer
from tqdm import tqdm

# Global variables
# Log bin for the noise distributions
bins = []
for dec in range(-3, 5, 1):
    bins.append(np.arange(10 ** dec, 10 ** (dec + 1), 10 ** dec))
bins = np.array(bins)
FREQ = bins.reshape(bins.shape[0] * bins.shape[1])

# Noise generation methods
NOISE_FREQ = 1     # Pixels have the same +/- noise frequency but with different phases
NOISE_MEASURE = 2  # Pixels have a noise distribution measured in one lighting conditions


class DvsSensor:
    """ Class to initialise and simulate the DVS sensor """
    # shape = (50, 50)                                    # Size of the imager
    # m_th_pos = 0.2                                      # Mean positive sensitivity (%)
    # m_th_neg = -0.2                                     # Mean negative sensitivity (%)
    # m_th_noise = 0.02                                   # Mean reset noise standard deviation of the transistor (%)
    # m_latency = 100                                     # Mean Latency (us)
    # tau = 100                                           # Time constant (us) of the logarithmic part at 1 klux
    # m_jitter = 30                                       # Mean jitter (us)
    # m_bgn_pos = 0.1                                     # Mean positive background frequency (Hz)
    # m_bgn_neg = 0.01                                    # Mean negative background frequency (Hz)
    # m_bgn_pos_per = np.uint64(1e6 / m_bgn_pos)          # Mean positive background period (us)
    # m_bgn_neg_per = np.uint64(1e6 / m_bgn_neg)          # Mean negative background period (us)
    # ref = 50                                            # Refractory period (us)
    # time = 0                                            # Time of the internal counter (us)
    # noise_model = NOISE_FREQ                            # Model of noise used
    # last_v       = np.zeros(shape, dtype=np.double)     # Voltage of each pixel during the last reset
    # cur_v        = np.zeros(shape, dtype=np.double)     # Voltage of each pixel at the time t
    # cur_th_pos   = np.zeros(shape, dtype=np.double)     # Current Positive Threshold
    # cur_th_neg   = np.zeros(shape, dtype=np.double)     # Current Negative Threshold
    # cur_ref      = np.zeros(shape, dtype=np.uint64)     # Time when the pixel will have to be reset
    # bgn_pos_next = np.zeros(shape, dtype=np.uint64)     # Next expected positive noise event
    # bgn_neg_next = np.zeros(shape, dtype=np.uint64)     # Next expected negative noise event
    # bgn_hist_pos = np.zeros((1, 1, 45), dtype=float)    # Positive noise cumulative distributions
    # bgn_hist_neg = np.zeros((1, 1, 45), dtype=float)    # Negative noise cumulative distributions
    # time_px      = np.zeros(shape, dtype=np.uint64)     # Time t at the pixel (us)
    # tau_p        = np.zeros(shape, dtype=np.double)     # Time constant of each pixel (us)

    def __init__(self, name):
        """ Init the sensor by creating the Blender Camera
        Args:
           name: string to identify the sensor
        """
        self.name = name

    def set_shape(self, x, y):
        """ Set the shape of the sensor
            Args:
                x, y: size of the image
        """
        self.shape = (x, y)

    def initCamera(self, x, y, lat, jit, ref, tau, th_pos, th_neg, th_noise, bgnp, bgnn):
        """ Set the properties of the DVS sensor

            In this version the sensor positive and negative event's properties are symmetrical
        Args:
              th_pos: Mean threshold (log change)
              th_neg: Mean threshold (log change)
              th_n: Threshold noise (log change)
              lat: asymptotic (infinite contrast) latency (us)
              tau: Time constant of the log conversion (us)
              jit: asymptotic jitter (us)
              bgn: Mean frequency of the noise (Hz)
        """
        self.shape = (x, y)
        self.m_th_pos = th_pos
        self.m_th_neg = -th_neg
        self.m_th_noise = th_noise
        self.m_latency = lat
        self.tau = tau
        self.m_jitter = jit
        self.m_bgn_pos = bgnp
        self.m_bgn_neg = bgnn
        self.m_bgn_pos_per = np.uint64(1e6 / bgnp)
        self.m_bgn_neg_per = np.uint64(1e6 / bgnn)
        self.ref = ref
        self.shape = (self.shape[1], self.shape[0])
        self.last_v = np.zeros(self.shape, dtype=np.double)
        self.cur_v = np.zeros(self.shape, dtype=np.double)
        self.cur_ref = np.zeros(self.shape, dtype=np.uint64)
        self.time_px = np.zeros(self.shape, dtype=np.uint64)
        self.tau_p = np.zeros(self.shape, dtype=np.double)
        self.cur_ref[:] = np.iinfo(np.uint64).max
        self.init_bgn()
        self.init_thresholds()
        self.time = 0

    def init_bgn(self):
        """ Initialise the phases of the background noise
            This noise model does not include noise differences between pixel: every pixel will fire noise events a the 
            same frequency but with a random phase
        """
        self.noise_model = NOISE_FREQ
        self.bgn_pos_next = np.array(np.random.randint(0, self.m_bgn_pos_per, self.shape), dtype=np.uint64)
        self.bgn_neg_next = np.array(np.random.randint(0, self.m_bgn_neg_per, self.shape), dtype=np.uint64)

    def init_bgn_hist(self, filename_noise_pos, filename_noise_neg):
        """ Load measured distributions of the noise,
            Pick randomly one noise distribution for each pixel and Initialise also randomly the phases of the
            background noise
            Args:
                filename_noise_pos: path of the positive noise's filename
                filename_noise_neg: path of the negative noise's filename
            """
        self.noise_model = NOISE_MEASURE
        noise_pos = np.load(filename_noise_pos)
        noise_pos = np.reshape(noise_pos, (noise_pos.shape[0] * noise_pos.shape[1], noise_pos.shape[2]))
        if len(noise_pos) == 0:
            print(filename_noise_pos, " is not correct")
            return
        noise_neg = np.load(filename_noise_neg)
        noise_neg = np.reshape(noise_neg, (noise_neg.shape[0] * noise_neg.shape[1], noise_neg.shape[2]))
        if len(noise_neg) == 0:
            print(filename_noise_neg, " is not correct")
            return
        
        # Load histogram and Update next noise event
        self.bgn_hist_pos = np.zeros((self.shape[0] * self.shape[1], 72), dtype=float)
        self.bgn_hist_neg = np.zeros((self.shape[0] * self.shape[1], 72), dtype=float)
        self.bgn_pos_next = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint64)
        self.bgn_neg_next = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint64)

        # Pick two spectra for each pixel (one for ON and one for OFF events)
        id_n = np.random.uniform(0, noise_neg.shape[0], size=(self.shape[0] * self.shape[1])).astype(int)
        id_p = np.random.uniform(0, noise_pos.shape[0], size=(self.shape[0] * self.shape[1])).astype(int)
        self.bgn_hist_pos = noise_pos[id_p, :]
        self.bgn_hist_neg = noise_neg[id_n, :]

        # Normalise noise spectra
        s_p = np.sum(self.bgn_hist_pos, axis=1)
        s_n = np.sum(self.bgn_hist_neg, axis=1)
        id_p = np.where(s_p == 0)
        self.bgn_hist_pos[id_p, 0] = 1
        id_n = np.where(s_n == 0)
        self.bgn_hist_neg[id_n, 0] = 1
        id_p = np.where(s_p > 0)
        self.bgn_hist_pos[id_p, :] = self.bgn_hist_pos[id_p, :] / \
                                     np.repeat(self.bgn_hist_pos[id_p, -2].reshape((1, id_p[0].shape[0], 1)),
                                               self.bgn_hist_pos.shape[1],
                                               axis=2)
        id_n = np.where(s_n > 0)
        self.bgn_hist_neg[id_n, :] = self.bgn_hist_neg[id_n, :] / \
                                     np.repeat(self.bgn_hist_neg[id_n, -2].reshape((1, id_n[0].shape[0], 1)),
                                               self.bgn_hist_neg.shape[1],
                                               axis=2)

        # Draw the next noise event time for each pixel
        for x in tqdm(range(0, self.shape[0], 1), desc="Noise Init"):
            for y in range(0, self.shape[1], 1):
                self.bgn_pos_next[x, y] = np.uint64(self.get_next_noise(x, y, 1) * np.random.uniform(0, 1))
                self.bgn_neg_next[x, y] = np.uint64(self.get_next_noise(x, y, 0) * np.random.uniform(0, 1))

    def init_thresholds(self):
        """ Initialise the thresholds of the comparators
            The positive and negative threshold share the same noise, which can be changed if necessary
        """
        self.cur_th_pos = np.clip(np.array(np.random.normal(self.m_th_pos, self.m_th_noise, self.shape),
                                           dtype=np.double), 0, 1000)
        self.cur_th_neg = np.clip(np.array(np.random.normal(self.m_th_neg, self.m_th_noise, self.shape),
                                           dtype=np.double), -1000, 0)

    def init_image(self, img):
        """ Initialise the first flux values of the sensor
        Args:
            img: image whose greylevel corresponds to a radiometric value
            It is assumed the maximum radiometric value is 1e6
        """
        if img.shape[1] != self.shape[1] or img.shape[0] != self.shape[0]:
            print("Error: the size of the image doesn't match with the sensor ")
            return
        self.last_v = np.log(img + 1)
        self.cur_v = np.log(img + 1)
        self.tau_p = self.tau * 1e3 / (img + 1)
        self.time_px[:, :] = 0
        self.time = 0

    def check_noise(self, dt, img_l):
        """ Generate event packet of noise
            Check if the time at each pixel crossed a next noise event threshold during the update
            In this method, every pixel has the same noise rate.
            Args:
                dt: delay between to images (us)
                img_l: log value of the input image
            Returns:
                A packet of events of type EventBuffer
        """
        ind_pos_noise = np.where(self.time + dt > self.bgn_pos_next)
        ind_neg_noise = np.where(self.time + dt > self.bgn_neg_next)
        pk_noise = EventBuffer(len(ind_pos_noise[0]) + len(ind_neg_noise[0]))
        if len(ind_pos_noise[0]) > 0:
            pk_noise.add_array(self.bgn_pos_next[ind_pos_noise], ind_pos_noise[0], ind_pos_noise[1], 1)
            self.time_px[ind_pos_noise] = self.bgn_pos_next[ind_pos_noise]
            self.bgn_pos_next[ind_pos_noise] += self.m_bgn_pos_per
            self.cur_v[ind_pos_noise] = img_l[ind_pos_noise]
            self.last_v[ind_pos_noise] = img_l[ind_pos_noise]
        if len(ind_neg_noise[0]) > 0:
            pk_noise.add_array(self.bgn_neg_next[ind_neg_noise], ind_neg_noise[0], ind_neg_noise[1], 0)
            self.time_px[ind_neg_noise] = self.bgn_neg_next[ind_neg_noise]
            self.bgn_neg_next[ind_neg_noise] += self.m_bgn_neg_per
            self.cur_v[ind_neg_noise] = img_l[ind_neg_noise]
            self.last_v[ind_neg_noise] = img_l[ind_neg_noise]
        pk_noise.sort()
        return pk_noise

    def check_noise_hist(self, dt, img_l):
        """ Generate event packet of noise
            Check if the time at each pixel crossed a next noise event threshold during the update
            This method uses a measured noise distribution for each pixel
            Args:
                  dt: delay between two updates (us)
                  img_l: logarithmic value of the input image
            Returns:
                A packet of events of type EventBuffer
        """
        ind_pos_noise = np.where(self.time + dt > self.bgn_pos_next)
        ind_neg_noise = np.where(self.time + dt > self.bgn_neg_next)
        pk_noise = EventBuffer(len(ind_pos_noise[0]) + len(ind_neg_noise[0]))
        if len(ind_pos_noise[0]) > 0:
            pk_noise.add_array(self.bgn_pos_next[ind_pos_noise], ind_pos_noise[0], ind_pos_noise[1], 1)
            self.time_px[ind_pos_noise] = self.bgn_pos_next[ind_pos_noise]
            self.cur_v[ind_pos_noise] = img_l[ind_pos_noise]
            self.last_v[ind_pos_noise] = img_l[ind_pos_noise]
            for i in range(0, len(ind_pos_noise[0]), 1):
                self.bgn_pos_next[ind_pos_noise[0][i], ind_pos_noise[1][i]] += \
                    self.get_next_noise(ind_pos_noise[1][i], ind_pos_noise[0][i], 1)
        if len(ind_neg_noise[0]) > 0:
            pk_noise.add_array(self.bgn_neg_next[ind_neg_noise], ind_neg_noise[0], ind_neg_noise[1], 0)
            self.time_px[ind_neg_noise] = self.bgn_neg_next[ind_neg_noise]
            self.cur_v[ind_neg_noise] = img_l[ind_neg_noise]
            self.last_v[ind_neg_noise] = img_l[ind_neg_noise]
            for i in range(0, len(ind_neg_noise[0]), 1):
                self.bgn_neg_next[ind_neg_noise[0][i], ind_neg_noise[1][i]] += \
                    self.get_next_noise(ind_neg_noise[1][i], ind_neg_noise[0][i], 0)
        pk_noise.sort()
        return pk_noise

    def get_next_noise(self, x, y, pol):
        """ Updates the next noise event
            Take a value between 0 and 1 and find the delay of the next noise event
            Args:
                x, y: coordinate of the pixel
                pol: polarity of the noise
            Returns:
                the delay of the next noise event in us
        """
        val = np.random.uniform(0, 1)
        pos = y * self.shape[0] + x
        if pol == 1:
            ind = np.where(self.bgn_hist_pos[pos, :] >= val)
            next = FREQ[ind[0][0]]
        else:
            ind = np.where(self.bgn_hist_neg[pos, :] >= val)
            next = FREQ[ind[0][0]]
        return np.uint64(1e6 / next)

    def get_latency(self, time_end, last_v, cur_th, cur_v, img_l, time_px):
        """ Obtain the latency of the pixel
            Method: Linearly interpolates the time when it crosses the threshold 
                    and add the constant latency of the comparator stage.
            Args:
                time_end: time of the change (us)
                last_v: voltage at the last spike (np.array)
                cur_th: threshold (np.array)
                cur_v: voltage at time_px (np.array)
                last_v: voltage during the last spike (np.array)
            Returns:
                np.array of the latencies in us
        """
        return np.uint64((last_v + cur_th - cur_v) / (img_l - cur_v) * (time_end - time_px) + \
               np.random.normal(self.m_latency, self.m_jitter, last_v.shape[0]))

    def get_latency_tau(self, cur_th, cur_v, img_l, tau_p):
        """ Obtain the latency of the pixel
            Method: First order low pass filter interpolation of the time when 
                    it crosses the threshold and add the constant latency of the
                    comparator stage
            Args:
                last_v: voltage at the last spike
                cur_th: threshold
                cur_v: voltage at time_px
                tau_p: time constants of the pixels
            Returns:
                np.array of the latencies in us
        """
        amp = np.divide(cur_th - cur_v, img_l - cur_v)
        jit = np.sqrt(self.m_jitter ** 2 + np.power(self.m_th_noise * tau_p / (img_l - cur_v), 2))
        t_ev = np.random.normal(self.m_latency - tau_p*np.log(1 - amp), jit)
        return np.uint64(np.clip(t_ev, 0, 10000))

    def update(self, img, dt):

        """ Update the sensor with a nef irradiance's frame
            Follow the ICNS model
            Args:
                img: radiometric value in the focal plane
                dt: delay between the frame and the last one (us)
            Returns:
                EventBuffer of the created events
             """
        if img.shape[1] != self.shape[1] or img.shape[0] != self.shape[0]:
            print("Error: the size of the image doesn't match with the sensor ")
            return
        
     
        # Convert in the log domain 0.003 s
        img_l = np.array(img, dtype=np.double)
        ind = np.where(img > 0)
        if len(ind[0]) == 0:
            print("ERROR: update: flux image with only zeros")
            # NOTE: If Arbiter is optimised this might need to done earlier
            #return
        img_l[ind] = np.log(img[ind] + 1)


        # Update time constants - self.tau defined at 1 klux
        self.tau_p[ind] = self.tau * 1e3 / (img[ind] + 1)
 
        # Update refractory and reset pixels - 0.001 s
        ind_ref = np.where(self.cur_ref < self.time + dt)
        px_delta_ref = np.array(self.cur_ref[ind_ref] - self.time, dtype=float)
        if len(ind_ref[0]) > 0:
            # Calculate voltage at the reset time (end of refractory period)
            self.last_v[ind_ref] = self.cur_v[ind_ref] + (img_l[ind_ref] - self.cur_v[ind_ref]) * \
                                (1 - np.exp(-px_delta_ref / self.tau_p[ind_ref]))
            # End the refractory period
            self.time_px[ind_ref] = self.cur_ref[ind_ref]
            self.cur_ref[ind_ref] = np.iinfo(np.uint64).max
            # And update the reference voltage for these pixels
            self.cur_v[ind_ref] = self.last_v[ind_ref]


        # Get noise events and reset pixels - 0.001 s
        if self.noise_model == NOISE_FREQ:
            pk_noise = self.check_noise(dt, img_l)
        else:
            pk_noise = self.check_noise_hist(dt, img_l)



        # Calculate voltage change at the end of the frame - 0.01 s
        px_delta_t = np.array(self.time + dt - self.time_px[ind], dtype=float)
        target = np.zeros(img_l.shape) 
        target[ind] = self.cur_v[ind] + (img_l[ind] - self.cur_v[ind]) * \
                    (1 - np.exp(-px_delta_t / self.tau_p[ind]))  
        dif = target - self.last_v


        # Check in which pixels the change is larger than the thresholds - 0.001 s
        ind_pos = np.where((dif > self.cur_th_pos) & (self.cur_ref == np.iinfo(np.uint64).max))
        ind_neg = np.where((dif < self.cur_th_neg) & (self.cur_ref == np.iinfo(np.uint64).max))


        # Generate events for these pixels - 0.03 s (low event - 0.002 s)
        pk = EventBuffer(0)
        while len(ind_pos[0]) + len(ind_neg[0]) > 0:
            pk.increase(len(ind_pos[0]) + len(ind_neg[0]))

            # ON events
            if len(ind_pos[0]) > 0:
                # Get event times
                # Use this for first order interpolation
                t_event = self.get_latency_tau(
                    self.last_v[ind_pos] + self.cur_th_pos[ind_pos], 
                    self.cur_v[ind_pos], 
                    img_l[ind_pos], 
                    self.tau_p[ind_pos]
                )
                # Or this for linear interpolation
                # t_event = self.get_latency(self.time + dt, 
                #                            self.last_v[ind_pos], 
                #                            self.cur_th_pos[ind_pos], 
                #                            self.cur_v[ind_pos], 
                #                            img_l[ind_pos], 
                #                            self.time_px[ind_pos]
                # )
                # Add to the event buffer
                pk.add_array(self.time_px[ind_pos] + t_event, ind_pos[0], ind_pos[1], 1)
                # Update the threshold with noise
                self.cur_th_pos[ind_pos] = np.clip(
                    np.random.normal(self.m_th_pos, self.m_th_noise, len(ind_pos[0])), 
                    0, 
                    1000
                )
                # Start the refractory period for those pixels that fired
                self.cur_ref[ind_pos] = self.time_px[ind_pos] + t_event + self.ref

            # OFF events
            if len(ind_neg[0]) > 0:
                # Get event times
                # Use this for first order interpolation
                t_event = self.get_latency_tau(
                    self.last_v[ind_neg] + self.cur_th_neg[ind_neg], 
                    self.cur_v[ind_neg], 
                    img_l[ind_neg], 
                    self.tau_p[ind_neg]
                )
                # Or this for linear interpolation
                # t_event = self.get_latency(self.time + dt, 
                #                            self.last_v[ind_neg], 
                #                            self.cur_th_neg[ind_neg], 
                #                            self.cur_v[ind_neg], 
                #                            img_l[ind_neg], 
                #                            self.time_px[ind_neg]
                # )
                # Add to the event buffer
                pk.add_array(self.time_px[ind_neg] + t_event, ind_neg[0], ind_neg[1], 0)
                # Update the threshold with noise
                self.cur_th_neg[ind_neg] = np.clip(
                    np.random.normal(self.m_th_neg, self.m_th_noise, len(ind_neg[0])), 
                    -1000, 
                    0
                )
                # Start the refractory period for those pixels that fired
                self.cur_ref[ind_neg] = self.time_px[ind_neg] + t_event + self.ref

            # Check if any of these refractory periods finish before the end of the frame
            ind_ref = np.where(self.cur_ref < self.time + dt)
            px_delta_ref = np.array(self.cur_ref[ind_ref]-self.time_px[ind_ref], dtype=float)
            if len(ind_ref[0]) > 0:
                # Calculate voltage at the reset time (end of refractory period)
                self.last_v[ind_ref] = self.cur_v[ind_ref] + (img_l[ind_ref] - self.cur_v[ind_ref]) * \
                                       (1 - np.exp(-px_delta_ref / self.tau_p[ind_ref]))
                # End the refractory period
                self.time_px[ind_ref] = self.cur_ref[ind_ref]
                self.cur_ref[ind_ref] = np.iinfo(np.uint64).max
                # And update the reference voltage for these pixels
                self.cur_v[ind_ref] = self.last_v[ind_ref]

            # Now check if there are any new threshold crossings since the previous event
            dif = np.zeros((self.shape[0], self.shape[1]))
            px_delta_ref = np.array(self.time + dt - self.time_px[ind_ref], dtype=float)
            target[ind_ref] = self.cur_v[ind_ref] + (img_l[ind_ref] - self.cur_v[ind_ref]) * \
                                (1 - np.exp(-px_delta_ref / self.tau_p[ind_ref]))
            dif[ind_ref] = target[ind_ref] - self.last_v[ind_ref]
            ind_pos = np.where(dif > self.cur_th_pos)
            ind_neg = np.where(dif < self.cur_th_neg)
            # Repeat this loop until no more threshold crossings are found
        

        # Update pixel voltages at end of frame - 0.007 s
        px_delta_t = np.array(self.time + dt - self.time_px[ind], dtype=float)
        self.cur_v[ind] = self.cur_v[ind] + (img_l[ind] - self.cur_v[ind]) * \
                                (1 - np.exp(-px_delta_t / self.tau_p[ind]))
        

        # Update simulation time 0.000 s
        self.time += dt
        self.time_px[:] = self.time


        # Merge noise and signal events and sort by time - 0.150 s (low event - 0.000 s)
        pk_end = EventBuffer(0)
        #pk_end.merge(pk, pk_noise) # - 0.120 s (low event - 0.000 s)



        # NOTE: merge_concat is faster than merge for large number of events
        pk_end.merge_concat(pk, pk_noise) # - 0.003 s (low event - 0.000 s)
        pk_end.sort() # - 0.002 s


        return pk_end



