import React from "react";
import Button from "./Button";
import { FaRegUserCircle } from "react-icons/fa";

const ButtonPreview: React.FC = () => {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        width: "400px",
        gap: "20px",
      }}
    >
      <Button color="default" size="large" isFullWidth>
        color="default" + size="large" + isFullWidth
      </Button>
      <Button color="inverse" size="large" isFullWidth>
        color="inverse" + size="large" + isFullWidth
      </Button>
      <Button color="dim" size="large" isFullWidth>
        color="dim" + size="large" + isFullWidth
      </Button>
      <Button color="green" size="large" isFullWidth>
        color="green" + size="large" + isFullWidth
      </Button>
      <Button color="red" size="large" isFullWidth>
        color="red" + size="large" + isFullWidth
      </Button>
      <Button
        color="default"
        size="large"
        isFullWidth
        leftIcon={<FaRegUserCircle size={52} />}
        rightIcon={<FaRegUserCircle size={52} />}
      >
        With iconssssssssss
      </Button>
    </div>
  );
};

export default ButtonPreview;
