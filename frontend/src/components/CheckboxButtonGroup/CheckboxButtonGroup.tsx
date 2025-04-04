import { CheckboxButton } from "../CheckboxButton/CheckboxButton";
import { CheckboxButtonGroupProps } from "./CheckboxButtonGroup.props";
import styles from "./CheckboxButtonGroup.module.css";

export const CheckboxButtonGroup = ({
  preferences,
}: CheckboxButtonGroupProps) => {
  return (
    <div className={styles.groupContainer}>
      {preferences.map((preference) => (
        <CheckboxButton
          key={preference.id}
          name={`preference-${preference.id}`}
        >
          {preference.value}
        </CheckboxButton>
      ))}
    </div>
  );
};
