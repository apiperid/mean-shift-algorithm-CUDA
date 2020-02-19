


CC = nvcc -O3 -lm -Wno-deprecated-gpu-targets

TARGETS = mean_shift_sequential mean_shift_without_shared_memory mean_shift_with_shared_memory


all: $(TARGETS)

mean_shift_sequential: mean_shift_sequential.c
	$(CC) $< -o $@

mean_shift_without_shared_memory: mean_shift_without_shared_memory.cu
	$(CC) $< -o $@

mean_shift_with_shared_memory: mean_shift_with_shared_memory.cu
	$(CC) $< -o $@

clean:
	$(RM) *.o *~ $(TARGETS)

