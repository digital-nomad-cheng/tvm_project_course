CONFIG  :=  ../config/Makefile.config

include $(CONFIG)

BUILD_PATH    :=  build
SRC_PATH      :=  src
CUDA_DIR      :=  /usr/local/cuda-$(CUDA_VER)

KERNELS_SRC   :=  $(wildcard $(SRC_PATH)/*.cu)

APP_OBJS      +=  $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(KERNELS_SRC:.cu=.cu.o))  

APP_DEPS      +=  $(KERNELS_SRC)

CUCC          :=  $(CUDA_DIR)/bin/nvcc

CUDAFLAGS     :=  -Xcompiler -fPIC 

INCS          :=  -I $(CUDA_DIR)/include \
                  -I $(SRC_PATH) 

LIBS          :=  -L "$(CUDA_DIR)/lib64" \

ifeq ($(DEBUG),1)
CUDAFLAGS     +=  -g -G -O0
else
CUDAFLAGS     +=  -O3
endif

ifeq ($(SHOW_WARNING),1)
CUDAFLAGS     +=  -Wall -Wunused-function -Wunused-variable -Wfatal-errors
else
CUDAFLAGS     +=  -w
endif

all:
	$(MAKE) $(APP)

update: $(APP)
	@echo finished updating $<

$(APP): $(APP_DEPS) $(APP_OBJS)
	@$(CUCC) $(APP_OBJS) -o $@ $(LIBS) $(INCS)
	@echo finished building $@. Have fun!!

show: 
	@echo $(BUILD_PATH)
	@echo $(APP_DEPS)
	@echo $(INCS)
	@echo $(APP_OBJS)

clean:
	rm -rf $(APP)
	rm -rf build

# Compile CUDA
$(BUILD_PATH)/%.cu.o: $(SRC_PATH)/%.cu
	@echo Compile CUDA $@
	@mkdir -p $(BUILD_PATH)
	@$(CUCC) -o $@ -c $< $(CUDAFLAGS) $(INCS)

.PHONY: all update show clean 
