class Logger:
    def __init__(self, writer, use_swanlab=False):
        self.writer = writer
        self.use_swanlab = use_swanlab

    def log(self, results, iter):
        # Prepare SwanLab log dict
        swanlab_dict = {}
        
        for key, val in results.items():
            if 'scalar' in key:
                # TensorBoard
                self.writer.add_scalar(key.replace('scalar/', ''), val, iter)
                # SwanLab - collect scalar values
                swanlab_key = key.replace('scalar/', '')
                swanlab_dict[swanlab_key] = val

            elif 'image' in key and 'images' not in key:
                # TensorBoard
                self.writer.add_image(key.replace('image/', ''), val, iter)
                # SwanLab - log images
                if self.use_swanlab:
                    try:
                        import swanlab
                        swanlab.log({key.replace('image/', ''): swanlab.Image(val)}, step=iter)
                    except:
                        pass

            elif 'images' in key:
                # TensorBoard
                self.writer.add_images(key.replace('images/', ''), val, iter)
                # SwanLab - log images
                if self.use_swanlab:
                    try:
                        import swanlab
                        swanlab.log({key.replace('images/', ''): swanlab.Image(val)}, step=iter)
                    except:
                        pass

            elif 'mesh' in key:
                # TensorBoard
                self.writer.add_mesh(key.replace('mesh/', ''), vertices=val['vertices'], colors=val['colors'], faces=val['faces'], global_step=iter)
                # Note: SwanLab doesn't support 3D mesh visualization directly
                # Skip mesh logging for SwanLab

        # Log all scalars to SwanLab at once
        if self.use_swanlab and swanlab_dict:
            try:
                import swanlab
                swanlab.log(swanlab_dict, step=iter)
            except Exception as e:
                # Silently fail if swanlab is not available or has issues
                pass
