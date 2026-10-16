from .dylibs import lcapi


class MeshFormat:
    # [
    #   [
    #       (lcapi.VertexAttributeType0, lcapi.PixelFormat0),
    #       (lcapi.VertexAttributeType1, lcapi.PixelFormat1)
    #   ],
    #   [
    #       (lcapi.VertexAttributeType0, lcapi.PixelFormat0),
    #       (lcapi.VertexAttributeType1, lcapi.PixelFormat1)
    #   ]
    # ]
    def __init__(self, streams: list):
        self.handle = lcapi.MeshFormat()
        for attributes in streams:
            for i in attributes:
                assert type(i[0]) == lcapi.VertexAttributeType and type(
                    i[1]) == lcapi.PixelFormat
                self.handle.add_attribute(i[0], i[1])
            self.handle.add_stream()
