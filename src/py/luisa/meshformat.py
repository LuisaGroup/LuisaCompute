import lcapi


class MeshFormat:
    # [
    #   [
    #       (lcapi.VertexAttributeType0, lcapi.VertexElementFormat0),
    #       (lcapi.VertexAttributeType1, lcapi.VertexElementFormat1)
    #   ],
    #   [
    #       (lcapi.VertexAttributeType0, lcapi.VertexElementFormat0),
    #       (lcapi.VertexAttributeType1, lcapi.VertexElementFormat1)
    #   ]
    # ]
    def __init__(self, streams: list):
        self.handle = lcapi.MeshFormat()
        for attributes in streams:
            for i in attributes:
                assert type(i[0]) == lcapi.VertexAttributeType and type(
                    i[1]) == lcapi.VertexElementFormat
                self.handle.add_attribute(i[0], i[1])
            self.handle.add_stream()
