clear; clc; close all;

% Simulation Parameters
Nzc = 63; % Zadoff-Chu sequence length
rootIdxvec = [8 23 53]; % Root indices for Zadoff-Chu sequences

carrierFreqSensing = 24e9; % Carrier frequency for sensing
carrierFreqComm = 3.5e9;   % Carrier frequency for communication
bandwidth = 100e6;         % Bandwidth
subcarrierSpacing = 30e3;  % Subcarrier spacing
numSubcarriers = floor(bandwidth / subcarrierSpacing);
fs = bandwidth;            % Sampling frequency
T_fmcw = 1e-6;             % FMCW chirp duration

numSymbols = 14; % OFDM symbols
cpLength = 16;  % Cyclic prefix length
modOrders = [4, 256, 1024]; % QPSK, 256-QAM, 1024-QAM
snrValues = -10:5:20; % SNR values
numIterations = 1e3;

% Multipath parameters
numPaths = 3;           % Number of multipath components
maxDelay = 5e-6;        % Maximum delay in seconds
gain_std = 1;   % Channel gain standard deviation

% Initialize result storage
cosSimZC_FMCW_avg = zeros(length(rootIdxvec), length(snrValues));
cosSimOFDM_FMCW_avg = zeros(length(modOrders), length(snrValues));

%% Generate FMCW Signal with Sufficient Length
signalLength = (numSubcarriers + cpLength) * numSymbols;
t = (0:1/fs:(signalLength-1)/fs).';
k = bandwidth / T_fmcw;
fmcwTx = exp(1i * 2 * pi * (carrierFreqSensing * t + 0.5 * k * t.^2));

for iter = 1:numIterations
    %% Zadoff-Chu + FMCW vs FMCW
    for i = 1:length(rootIdxvec)
        % Zadoff-Chu Sequence (Frequency -> Time)
        zcFreq = zadoffChuSeq(rootIdxvec(i), Nzc);
        zcTime = ifft(zcFreq, Nzc, 'symmetric');
        zcTime_cp = [zcTime(end-cpLength+1:end); zcTime]; % CP 추가
        
        % Adjust FMCW length
        fmcwTx_trim = fmcwTx(1:length(zcTime_cp));
        
        % Combine with FMCW
        zc_fmcw_combined = (fmcwTx_trim + zcTime_cp)/norm(fmcwTx_trim + zcTime_cp);
        
        % Apply multipath delay and gain
        multipath_zc = zeros(size(zc_fmcw_combined));
        for p = 1:numPaths
            delaySamples = min(round(rand * maxDelay * fs), length(zc_fmcw_combined) - 1);
            pathGain = (randn + 1i * randn) * gain_std;
            delayedSignal = [zeros(delaySamples, 1); zc_fmcw_combined(1:end-delaySamples)]* pathGain;
            multipath_zc = multipath_zc(1:length(delayedSignal)) + delayedSignal;
        end
        
        % Upconversion
        tUp = (0:length(multipath_zc)-1).' / fs;
        multipath_zc_up = multipath_zc .* exp(1i * 2 * pi * carrierFreqComm * tUp);
        fmcwTx_up = fmcwTx_trim .* exp(1i * 2 * pi * carrierFreqSensing * tUp);

        for idx = 1:length(snrValues)
            snr = snrValues(idx);
            
            % Add noise
            zc_fmcw_noisy = awgn(multipath_zc_up, snr, 'measured');
            fmcw_noisy = awgn(fmcwTx_up, snr, 'measured');
            
            % Downconversion
            zc_fmcw_down = zc_fmcw_noisy .* exp(-1i * 2 * pi * carrierFreqComm * tUp);
            fmcw_down = fmcw_noisy .* exp(-1i * 2 * pi * carrierFreqSensing * tUp);
            
            % remove CP
            zc_fmcw_noisy_cp_removed = zc_fmcw_down(cpLength+1:end);
            fmcw_noisy_cp_removed = fmcw_down(cpLength+1:end);

            % Cosine similarity
            minLen = min(length(zc_fmcw_noisy_cp_removed), length(fmcw_noisy_cp_removed));
            zc_fmcw_noisy_cp_removed = fft(zc_fmcw_noisy_cp_removed(1:minLen));
            fmcw_noisy_cp_removed = fft(fmcw_noisy_cp_removed(1:minLen));

            cosSimZC_FMCW_avg(i, idx) = cosSimZC_FMCW_avg(i, idx) + ...
                abs(sum(zc_fmcw_noisy_cp_removed .* conj(fmcw_noisy_cp_removed))) / ...
                (norm(zc_fmcw_noisy_cp_removed) * norm(fmcw_noisy_cp_removed));
        end
    end

    %% OFDM + FMCW vs FMCW
    for modIdx = 1:length(modOrders)
        modOrder = modOrders(modIdx);

        % Generate OFDM Signal
        numDataSubcarriers = numSubcarriers - 12;
        numTotalSymbols = numDataSubcarriers * numSymbols;
        data = randi([0 modOrder-1], numTotalSymbols, 1);
        modData = qammod(data, modOrder, 'UnitAveragePower', true);
        modData = reshape(modData, numDataSubcarriers, numSymbols);

        txOFDM = ifft([zeros(6, numSymbols); modData; zeros(6, numSymbols)], numSubcarriers); % IFFT
        txOFDM_cp = [txOFDM(end-cpLength+1:end, :); txOFDM]; % CP 추가
        txOFDM_serial = txOFDM_cp(:); % Serialize
        
        % Adjust FMCW length
        fmcwTx_trim = fmcwTx(1:length(txOFDM_serial));
        
        % Combine with FMCW
        ofdm_fmcw_combined = (fmcwTx_trim + txOFDM_serial)/norm(fmcwTx_trim + txOFDM_serial);
        
        % Apply multipath delay and gain
        multipath_ofdm = zeros(size(ofdm_fmcw_combined));
        for p = 1:numPaths
            delaySamples = min(round(rand * maxDelay * fs), length(ofdm_fmcw_combined) - 1);
            pathGain = (randn + 1i * randn) * gain_std;
            delayedSignal = [zeros(delaySamples, 1); ofdm_fmcw_combined(1:end-delaySamples)]* pathGain;
            multipath_ofdm = multipath_ofdm(1:length(delayedSignal)) + delayedSignal;
        end
        
        % Upconversion
        tUp = (0:length(multipath_ofdm)-1).' / fs;
        multipath_ofdm_up = multipath_ofdm .* exp(1i * 2 * pi * carrierFreqComm * tUp);
        fmcwTx_up = fmcwTx_trim .* exp(1i * 2 * pi * carrierFreqSensing * tUp);

        for idx = 1:length(snrValues)
            snr = snrValues(idx);

            % Add noise
            ofdm_fmcw_noisy = awgn(multipath_ofdm_up, snr, 'measured');
            fmcw_noisy = awgn(fmcwTx_up, snr, 'measured');
            
            % Downconversion
            ofdm_fmcw_down = ofdm_fmcw_noisy .* exp(-1i * 2 * pi * carrierFreqComm * tUp);
            fmcw_down = fmcw_noisy .* exp(-1i * 2 * pi * carrierFreqSensing * tUp);

            % remove CP
            ofdm_fmcw_noisy_cp_removed = ofdm_fmcw_down(cpLength+1:end);
            fmcw_noisy_cp_removed = fmcw_down(cpLength+1:end);

            % Cosine similarity
            minLen = min(length(ofdm_fmcw_noisy_cp_removed), length(fmcw_noisy_cp_removed));
            ofdm_fmcw_noisy_cp_removed = fft(ofdm_fmcw_noisy_cp_removed(1:minLen));
            fmcw_noisy_cp_removed = fft(fmcw_noisy_cp_removed(1:minLen));

            cosSimOFDM_FMCW_avg(modIdx, idx) = cosSimOFDM_FMCW_avg(modIdx, idx) + ...
                abs(sum(ofdm_fmcw_noisy_cp_removed .* conj(fmcw_noisy_cp_removed))) / ...
                (norm(ofdm_fmcw_noisy_cp_removed) * norm(fmcw_noisy_cp_removed));
        end
    end
end

% Average results
cosSimZC_FMCW_avg = cosSimZC_FMCW_avg / numIterations;
cosSimOFDM_FMCW_avg = cosSimOFDM_FMCW_avg / numIterations;

%% Plot results
figure(1);
for i = 1:length(rootIdxvec)
    semilogy(snrValues, cosSimZC_FMCW_avg(i, :), '-o', 'DisplayName', ...
        sprintf('Zadoff-Chu + FMCW vs FMCW (Root index: %d)', rootIdxvec(i)));
    hold on;
end
xlabel('SNR [dB]'); ylabel('|Cosine Similarity|');
legend; grid on;

figure(1);
for modIdx = 1:length(modOrders)
    semilogy(snrValues, cosSimOFDM_FMCW_avg(modIdx, :), '-o', 'DisplayName', ...
        sprintf('OFDM + FMCW vs FMCW (%d-QAM)', modOrders(modIdx)));
    hold on;
end
legend; grid on;




