import { CheckboxButtonProps } from "./CheckboxButton.props";
import styles from "./CheckboxButton.module.css";

export const CheckboxButton = ({ name, children }: CheckboxButtonProps) => {
  return (
    <div className={styles.checkboxContainer}>
      <input type="checkbox" id={name} name={name} className={styles.input} />
      <label htmlFor={name} className={styles.label}>
        {children}
      </label>
    </div>
  );
};
