clear; clc; close all;

% Simulation Parameters
Nzc = 63; % Zadoff-Chu sequence length
rootIdxvec = [8, 23, 53]; % Root indices for Zadoff-Chu sequences
carrierFreqSensing = 24e9; % Carrier frequency for FMCW
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
numIterations = 1e2;

% Initialize result storage
cosSimZC_FMCW_avg = zeros(length(rootIdxvec), length(snrValues));
cosSimOFDM_FMCW_avg = zeros(length(modOrders), length(snrValues));

%% Generate FMCW Signal
signalLength = (numSubcarriers + cpLength) * numSymbols;
t = (0:1/fs:(signalLength-1)/fs).';
k = bandwidth / T_fmcw;
fmcwTx = exp(1i * 2 * pi * (carrierFreqSensing * t + 0.5 * k * t.^2));

for iter = 1:numIterations
    %% Zadoff-Chu + FMCW vs FMCW
    for i = 1:length(rootIdxvec)
        % Zadoff-Chu Sequence
        zcFreq = zadoffChuSeq(rootIdxvec(i), Nzc);
        zcTime = ifft(zcFreq, Nzc, 'symmetric');
        zcTime_cp = [zcTime(end-cpLength+1:end); zcTime]; % CP 추가

        % Adjust FMCW length
        fmcwTx_trim = fmcwTx(1:length(zcTime_cp));
        zc_fmcw_combined = (fmcwTx_trim + zcTime_cp)/norm(fmcwTx_trim + zcTime_cp);

        % Upconversion
        tUp = (0:length(zc_fmcw_combined)-1).' / fs;
        zc_fmcw_up = zc_fmcw_combined .* exp(1i * 2 * pi * carrierFreqComm * tUp);
        fmcwTx_up = fmcwTx_trim .* exp(1i * 2 * pi * carrierFreqSensing * tUp);

        for idx = 1:length(snrValues)
            snr = snrValues(idx);

            % Add AWGN noise
            zc_fmcw_noisy = awgn(zc_fmcw_up, snr, 'measured');
            fmcw_noisy = awgn(fmcwTx_up, snr, 'measured');

            % Downconversion
            zc_fmcw_down = zc_fmcw_noisy .* exp(-1i * 2 * pi * carrierFreqComm * tUp);
            fmcw_down = fmcw_noisy .* exp(-1i * 2 * pi * carrierFreqSensing * tUp);

            % Remove CP
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

        txOFDM = ifft([zeros(6, numSymbols); modData; zeros(6, numSymbols)], numSubcarriers);
        txOFDM_cp = [txOFDM(end-cpLength+1:end, :); txOFDM];
        txOFDM_serial = txOFDM_cp(:);
        
        
        % Adjust FMCW length
        fmcwTx_trim = fmcwTx(1:length(txOFDM_serial));
        ofdm_fmcw_combined = (fmcwTx_trim + txOFDM_serial)/norm(fmcwTx_trim + txOFDM_serial);

        % Upconversion
        tUp = (0:length(ofdm_fmcw_combined)-1).' / fs;
        ofdm_fmcw_up = ofdm_fmcw_combined .* exp(1i * 2 * pi * carrierFreqComm * tUp);
        fmcwTx_up = fmcwTx_trim .* exp(1i * 2 * pi * carrierFreqSensing * tUp);

        for idx = 1:length(snrValues)
            snr = snrValues(idx);

            % Add AWGN noise
            ofdm_fmcw_noisy = awgn(ofdm_fmcw_up, snr, 'measured');
            fmcw_noisy = awgn(fmcwTx_up, snr, 'measured');

            % Downconversion
            ofdm_fmcw_down = ofdm_fmcw_noisy .* exp(-1i * 2 * pi * carrierFreqComm * tUp);
            fmcw_down = fmcw_noisy .* exp(-1i * 2 * pi * carrierFreqSensing * tUp);

            % Remove CP
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
%title('Zadoff-Chu + FMCW vs FMCW');
legend; grid on;

figure(1);
for modIdx = 1:length(modOrders)
    semilogy(snrValues, cosSimOFDM_FMCW_avg(modIdx, :), '-o', 'DisplayName', ...
        sprintf('OFDM + FMCW vs FMCW (%d-QAM)', modOrders(modIdx)));
    hold on;
end
% xlabel('SNR [dB]'); ylabel('|Cosine Similarity|');
% title('OFDM + FMCW vs FMCW');
legend; grid on;
