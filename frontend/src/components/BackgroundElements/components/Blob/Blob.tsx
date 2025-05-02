import { useRef } from "react";
import * as blobs2 from "blobs/v2";

import { BlobProps } from "./Blob.props";

const Blob: React.FC<BlobProps> = ({
  color = "var(--solitude-100)",
  seed,
  extraPoints = 8,
  randomness = 4,
  size = 256,
}) => {
  const stableSeedRef = useRef(seed ?? Math.random());

  const pathData = blobs2.svgPath({
    seed: stableSeedRef.current,
    extraPoints: extraPoints,
    randomness: randomness,
    size: size,
  });

  return (
    <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size}>
      <path d={pathData} fill={color} />
    </svg>
  );
};

export default Blob;
