import classes from "./Example.module.css";
import { ExampleProps } from "./Example.props";

const Example: React.FC<ExampleProps> = ({ text, children }) => {
  // Также можно писать особые классы с помощью classnames - https://www.npmjs.com/package/classnames

  return (
    <>
      <div className={classes.div1}>{text}</div>
      {children && <div className={classes.div2}>Ребенок - {children}</div>}
    </>
  );
};

export default Example;
