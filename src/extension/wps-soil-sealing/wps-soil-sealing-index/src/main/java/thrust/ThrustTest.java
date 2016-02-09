package thrust;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include={"<thrust/host_vector.h>", "<thrust/device_vector.h>", "<thrust/generate.h>", "<thrust/sort.h>",
                   "<thrust/copy.h>", "<thrust/reduce.h>", "<thrust/functional.h>", "<algorithm>", "<cstdlib>"})
@Namespace("thrust")
public class ThrustTest {
    static { Loader.load(); }

    public static class IntGenerator extends FunctionPointer {
        static { Loader.load(); }
        protected IntGenerator() { allocate(); }
        private native void allocate();
        public native int call();
    }

    @Name("plus<int>")
    public static class IntPlus extends Pointer {
        static { Loader.load(); }
        public IntPlus() { allocate(); }
        private native void allocate();
        public native @Name("operator()") int call(int x, int y);
    }

    @Name("host_vector<int>")
    public static class IntHostVector extends Pointer {
        static { Loader.load(); }
        public IntHostVector() { allocate(0); }
        public IntHostVector(long n) { allocate(n); }
        public IntHostVector(IntDeviceVector v) { allocate(v); }
        private native void allocate(long n);
        private native void allocate(@ByRef IntDeviceVector v);

        public IntPointer begin() { return data(); }
        public IntPointer end() { return data().position((int)size()); }

        public native IntPointer data();
        public native long size();
        public native void resize(long n);
    }

    @Name("device_ptr<int>")
    public static class IntDevicePointer extends Pointer {
        static { Loader.load(); }
        public IntDevicePointer() { allocate(null); }
        public IntDevicePointer(IntPointer ptr) { allocate(ptr); }
        private native void allocate(IntPointer ptr);

        public native IntPointer get();
    }

    @Name("device_vector<int>")
    public static class IntDeviceVector extends Pointer {
        static { Loader.load(); }
        public IntDeviceVector() { allocate(0); }
        public IntDeviceVector(long n) { allocate(n); }
        public IntDeviceVector(IntHostVector v) { allocate(v); }
        private native void allocate(long n);
        private native void allocate(@ByRef IntHostVector v);

        public IntDevicePointer begin() { return data(); }
        public IntDevicePointer end() { return new IntDevicePointer(data().get().position((int)size())); }

        public native @ByVal IntDevicePointer data();
        public native long size();
        public native void resize(long n);
    }

    public static native @MemberGetter @Namespace IntGenerator rand();
    public static native void copy(@ByVal IntDevicePointer first, @ByVal IntDevicePointer last, IntPointer result);
    public static native void generate(IntPointer first, IntPointer last, IntGenerator gen);
    public static native void sort(@ByVal IntDevicePointer first, @ByVal IntDevicePointer last);
    public static native int reduce(@ByVal IntDevicePointer first, @ByVal IntDevicePointer last, int init, @ByVal IntPlus binary_op);

    public static void main(String[] args) {
        // generate 32M random numbers serially
        IntHostVector h_vec = new IntHostVector(32 << 20);
        generate(h_vec.begin(), h_vec.end(), rand());

        // transfer data to the device
        IntDeviceVector d_vec = new IntDeviceVector(h_vec);

        // sort data on the device (846M keys per second on GeForce GTX 480)
        sort(d_vec.begin(), d_vec.end());

        // transfer data back to host
        copy(d_vec.begin(), d_vec.end(), h_vec.begin());

        // compute sum on device
        int x = reduce(d_vec.begin(), d_vec.end(), 0, new IntPlus());
    }
}
