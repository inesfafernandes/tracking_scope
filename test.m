npts=16
step = cos(linspace(0,2*pi,npts)');
outputSignal1 = sin(linspace(0,2*pi,npts)');
%%
hold on
plot(step)
plot(outputSignal1)
hold off
%%
conv_sig=conv(step,outputSignal1,'same');
fft_step=fft(step);
fft_signal=fft(outputSignal1);
mult_fft=fft_step.*fft_signal;
ifft_mult=ifft(mult_fft);

figure(1)
plot(conv_sig);
figure(2)
plot(ifft_mult);