import { components } from "react-select";
import { FaCheck } from "react-icons/fa6"; 
import classes from "./CustomOption.module.css";

const CustomOption = (props: any) => {
  const { isSelected, label } = props;

  return (
    <div title={label}>
      <components.Option {...props}>
        <div style={{ 
          display: "flex", 
          alignItems: "center", 
          justifyContent: "space-between", 
          width: "100%",
          gap: "8px" 
        }}>
          <span className={classes.optionText}>{label}</span>
          {isSelected && <FaCheck size={18} style={{ marginLeft: "auto" }}/>}
        </div>
      </components.Option>
    </div>
  );
};

export default CustomOption;